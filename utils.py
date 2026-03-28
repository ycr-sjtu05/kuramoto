import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import DataLoader

# ==========================================
# 1. 数学空间映射工具 (像素空间 <-> 相位空间)
# ==========================================

def map_to_phase(x):
    """
    将图像像素映射到相位空间。
    假设输入 x 的范围是 [-1, 1]，输出范围将严格控制在 [-0.9pi, 0.9pi]。
    这样可以为后续的 Kuramoto 物理演化留出一定的拓扑缓冲空间。
    """
    return x * 0.9 * torch.pi

def phase_modulate(x):
    """
    相位调制函数：将实数轴上任意大的角度强行折叠回 [-pi, pi] 主值区间。
    全量使用 PyTorch 原生算子，杜绝 np.pi 带来的隐式设备/类型转换隐患。
    """
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

def map_to_image(phases):
    """
    将网络生成的相位还原回 [-1, 1] 的图像像素。
    用于验证出图 (train.py) 和特征计算 (fid.py)。
    """
    images = phases / (0.9 * torch.pi)
    # 严格截断到 [-1, 1]，防止极少数未完全收敛的异常点导致整张生成的网格图过曝或发黑
    return images.clamp(-1.0, 1.0)


# ==========================================
# 2. 原始数据集加载器
# ==========================================

def get_data(args):
    """
    获取原始图像数据集的 DataLoader。
    输出的图像 Tensor 值域将被严格归一化到 [-1, 1]。
    """
    # 基础的 Transform 流程
    transform_list = [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ]
    
    if args.dataset_name == "cifar":
        # CIFAR-10 是 3 通道 RGB
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        dataset = CIFAR10(
            root=args.dataset_path, 
            train=True, 
            download=True, 
            transform=transforms.Compose(transform_list)
        )
        
    elif args.dataset_name == "fp":
        # SOCOFing 指纹数据集通常是单通道灰度图
        # 强制转换为单通道，防止部分原始图片是伪彩色图
        transform_list.insert(1, transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        
        # 假设你的 SOCOFing 数据存放在 args.dataset_path 下
        # 注意：ImageFolder 要求数据目录下必须至少有一层子文件夹 (比如 /data/raw/Real/... 和 /data/raw/Altered/...)
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"找不到数据集路径: {args.dataset_path}")
            
        dataset = ImageFolder(
            root=args.dataset_path, 
            transform=transforms.Compose(transform_list)
        )
        
    elif args.dataset_name == "alot":
        # ALOT 是 3 通道 RGB 图像
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"找不到数据集路径: {args.dataset_path}")

        dataset = ImageFolder(
            root=args.dataset_path,
            transform=transforms.Compose(transform_list)
        )
        
        
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset_name}")

    # 获取 num_workers，兼容没有在 args 里设置的情况
    num_workers = getattr(args, 'num_workers', 4)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True # 开启 pin_memory 加速数据往 GPU 的搬运
    )
    
    return dataloader, dataset
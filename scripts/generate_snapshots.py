import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils import get_data
# 从原版代码中提取的辅助函数
def map_to_phase(x):
    return x * 0.9 * np.pi

def phase_modulate(x):
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi


class KuramotoSimulator:
    """
    纯物理的离线 Kuramoto 模拟器。
    我们彻底剥离了高斯噪声，只保留纯确定性的拓扑同步过程。
    """
    def __init__(self, steps=200, 
                 k_start=5e-5, k_end=0.03, 
                 k_ref_start=5e-5, k_ref_end=0.03, 
                 ref_phase=0.0, M=5, device="cuda"):
        self.steps = steps
        self.device = device
        self.ref_phase = ref_phase
        self.M = M  # 局部耦合的窗口半径，M=5 对应 11x11 的窗口
        
        # 分离局部耦合系数 K(t) 和 全局参考耦合系数 K_ref(t)
        self.k_t = torch.linspace(k_start, k_end, steps).to(device)
        self.k_ref_t = torch.linspace(k_ref_start, k_ref_end, steps).to(device)
        self.kernel = None

    
    def compute_drift(self, x, t):
        """
        计算单步的确定性物理受力 (Drift)
        完美等价于论文公式: K(t)/N * \sum sin(\theta_j - \theta_i) + K_ref(t) * sin(\psi_ref - \theta_i)
        """
        C = x.shape[1]  # 通道数
        
        # 1. 计算局部耦合 Drift (修正了原版直接平均角度的 Bug)
        # 用欧拉公式 e^(i\theta) 映射到复平面
        cos_x, sin_x = torch.cos(x), torch.sin(x)
        
        # ==========================================
        # 🚀 性能优化核心：懒加载 (Lazy Loading)
        # 只在第一次运行时（或通道数发生突变时）才在 GPU 上申请显存生成 kernel，
        # 随后的 199 步全部直接复用内存！
        # ==========================================
        if not hasattr(self, 'kernel') or self.kernel is None or self.kernel.shape[0] != C:
            kernel_size = 2 * self.M + 1
            base_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device) / (kernel_size ** 2)
            self.kernel = base_kernel.repeat(C, 1, 1, 1)
        
        # 周期性 Padding，保证图像边缘的拓扑连贯性
        cos_padded = F.pad(cos_x, (self.M, self.M, self.M, self.M), mode='circular')
        sin_padded = F.pad(sin_x, (self.M, self.M, self.M, self.M), mode='circular')
        
        # 在复平面上求局部质心 (Mean Field)，直接使用我们缓存好的 self.kernel
        mean_cos = F.conv2d(cos_padded, self.kernel, groups=C)
        mean_sin = F.conv2d(sin_padded, self.kernel, groups=C)
        
        # 局部受力公式 R * sin(Phi - x) = R * sin(Phi)cos(x) - R * cos(Phi)sin(x)
        local_drift = mean_sin * cos_x - mean_cos * sin_x
        
        # 2. 计算全局参考 Drift
        global_drift = torch.sin(self.ref_phase - x)
        
        # 3. 组装总 Drift
        total_drift = self.k_t[t] * local_drift + self.k_ref_t[t] * global_drift
        
        return total_drift


def generate_offline_snapshots(args):
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 1. 加载数据
    train_dataloader, _ = get_data(args)
    
    # 2. 实例化模拟器
    simulator = KuramotoSimulator(
        steps=args.sim_steps,
        k_start=args.k_start, k_end=args.k_end,
        k_ref_start=args.k_ref_start, k_ref_end=args.k_ref_end,
        ref_phase=args.ref_phase,
        M=args.window_radius,
        device=args.device
    )
    
    print(f"开始离线生成快照... 共 {args.sim_steps} 步物理演化。")
    print(f"每 {args.save_interval} 步保存一次，输出路径: {args.save_dir}")
    
    img_counter = 0
    for batch_idx, (images, _) in enumerate(tqdm(train_dataloader)):
        images = images.to(args.device)
        
        # 将像素 [-1, 1] 映射到初始相位 [-0.9pi, 0.9pi]
        x = map_to_phase(images)
        
        # 保存轨迹的列表，A_0 是未改变的初始图
        snapshots = [x.clone().cpu().numpy()]
        
        # 3. 运行纯物理 ODE
        for t in range(args.sim_steps):
            # 获取当前步的物理速度
            drift = simulator.compute_drift(x, t)
            
            # 欧拉步进 (这里相当于 dt=1)
            x = x + drift
            
            # 将跑飞的角度拉回 [-pi, pi] 主值区间
            x = phase_modulate(x)
            
            # 记录快照点
            if (t + 1) % args.save_interval == 0:
                snapshots.append(x.clone().cpu().numpy())
                
        # 拼装成 Numpy 数组: Shape -> (N_snapshots, Batch, C, H, W)
        snapshots_np = np.stack(snapshots, axis=0)
        
        # ==========================================
        # 核心：时间轴 Unwrap，将圆环降维到实数轴
        # ==========================================
        unwrapped_snapshots_np = np.unwrap(snapshots_np, axis=0)
        
        # 转回 Tensor
        unwrapped_snapshots = torch.from_numpy(unwrapped_snapshots_np).float()
        
        # 4. 单张图分离并落盘
        # shape: (N_snapshots, B, C, H, W) -> 取单图 (N_snapshots, C, H, W)
        B = unwrapped_snapshots.shape[1]
        for b in range(B):
            single_img_track = unwrapped_snapshots[:, b, :, :, :].clone()
            save_path = os.path.join(args.save_dir, f"{img_counter:06d}.pt")
            torch.save(single_img_track, save_path)
            img_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Kuramoto Snapshot Generator")
    
    # 数据集配置
    parser.add_argument("--dataset_name", type=str, default="fp", help="fp (SOCOFing), cifar, tex")
    parser.add_argument("--dataset_path", type=str, default="./data/raw")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟与保存配置
    parser.add_argument("--sim_steps", type=int, default=200, help="总演化步数")
    parser.add_argument("--save_interval", type=int, default=20, help="每隔多少步保存一张快照")
    parser.add_argument("--save_dir", type=str, default="./data/snapshots/kura_snapshots")
    
    # 物理超参数 (默认对齐原文的 Local Coupling)
    parser.add_argument("--k_start", type=float, default=5e-5)
    parser.add_argument("--k_end", type=float, default=0.03)
    parser.add_argument("--k_ref_start", type=float, default=5e-5)
    parser.add_argument("--k_ref_end", type=float, default=0.03)
    parser.add_argument("--ref_phase", type=float, default=0.0)
    parser.add_argument("--window_radius", type=int, default=5, help="局部耦合的窗口半径 M (窗口大小为 2M+1)")
    
    args = parser.parse_args()
    generate_offline_snapshots(args)
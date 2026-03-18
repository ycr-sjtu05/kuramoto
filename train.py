import os
import copy
import torch
import wandb
import argparse
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

# 导入我们全新编写的核心模块
from datasets.snapshot_data import SnapshotDataset
from core.interpolant_lib import InterpolantEngine
from core.losses import DecoupledInterpolantLoss
from models.network import PhaseUNet, EMA
from sample import KuramotoSampler, DualHeadModel

# 从原版引入一些辅助功能 (如果你把它们放在 utils.py 里的话)
from utils import map_to_image
import torchvision

def train_interpolant(args):
    device = args.device
    
    # 1. 初始化 WandB 实验追踪
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    # 创建本地 checkpoint 目录
    ckpt_dir = os.path.join("experiments", args.run_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 2. 挂载极速离线快照数据流
    print(f"Loading snapshots from: {args.snapshot_dir}")
    dataset = SnapshotDataset(args.snapshot_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    # 3. 实例化数学引擎与 Loss
    engine = InterpolantEngine(num_snapshots=args.num_snapshots, sigma_max=args.sigma_max, device=device)
    criterion = DecoupledInterpolantLoss(interpolant_engine=engine, lambda_E=args.lambda_E)
    
    # 4. 实例化双头网络 (D: 结构漂移, E: 高频去噪)
    print("Initializing Model D (Drift) and Model E (Noise)...")
    model_D = PhaseUNet(c_in=args.channels, c_out=args.channels, img_size=args.image_size, device=device).to(device)
    model_E = PhaseUNet(c_in=args.channels, c_out=args.channels, img_size=args.image_size, device=device).to(device)
    
    # 分配独立优化器
    opt_D = torch.optim.AdamW(model_D.parameters(), lr=args.lr)
    opt_E = torch.optim.AdamW(model_E.parameters(), lr=args.lr)
    
    # 设置 EMA (指数滑动平均)，这对于生成模型的最终质量至关重要
    ema_D = EMA(0.995)
    ema_E = EMA(0.995)
    ema_model_D = copy.deepcopy(model_D).eval().requires_grad_(False)
    ema_model_E = copy.deepcopy(model_E).eval().requires_grad_(False)

    # 5. 实例化采样器 (用于中途验证出图)
    # 我们用步数较少的 ODE 来快速预览效果
    sampler = KuramotoSampler(engine, img_size=args.image_size, channels=args.channels, steps=50, noise_scale=0.0, device=device)

    # ==========================================
    # 开始训练大循环
    # ==========================================
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model_D.train()
        model_E.train()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, snapshots in enumerate(pbar):
            snapshots = snapshots.to(device).float()
            
            # 核心：一行代码算完所有的数学投影和 Loss！
            loss, loss_dict = criterion(model_D, model_E, snapshots)
            
            # 反向传播与参数更新
            opt_D.zero_grad()
            opt_E.zero_grad()
            loss.backward()
            opt_D.step()
            opt_E.step()
            
            # 更新 EMA 权重
            ema_D.step_ema(ema_model_D, model_D)
            ema_E.step_ema(ema_model_E, model_E)
            
            # 记录日志到 WandB
            wandb.log(loss_dict, step=global_step)
            
            pbar.set_postfix(Loss=f"{loss.item():.4f}", D=f"{loss_dict['loss_D_drift']:.4f}", E=f"{loss_dict['loss_E_noise']:.4f}")
            global_step += 1
            
        # ==========================================
        # 验证与出图 (每隔 eval_freq 个 Epoch)
        # ==========================================
        if epoch % args.eval_freq == 0:
            print(f"Generating sample images at Epoch {epoch}...")
            # 把 EMA 网络打包喂给采样器
            dual_model_ema = DualHeadModel(ema_model_D, ema_model_E)
            
            # 生成 16 张图看看效果
            sampled_phases = sampler.sample(dual_model_ema, n=16)
            
            # 映射回 RGB/灰度像素
            sampled_images = map_to_image(sampled_phases)
            
            # 拼成 4x4 的大图网格
            grid = torchvision.utils.make_grid(sampled_images, nrow=4)
            
            # 直接把图片发送给 WandB (硬盘里都不用存了！)
            wandb.log({"Generated Samples": wandb.Image(grid, caption=f"Epoch {epoch}")}, step=global_step)
            
        # ==========================================
        # 保存 Checkpoint
        # ==========================================
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_state = {
                'epoch': epoch,
                'global_step': global_step,
                'model_D': model_D.state_dict(),
                'model_E': model_E.state_dict(),
                'ema_model_D': ema_model_D.state_dict(),
                'ema_model_E': ema_model_E.state_dict(),
                'opt_D': opt_D.state_dict(),
                'opt_E': opt_E.state_dict(),
            }
            # 保存 latest 和 specific epoch
            torch.save(ckpt_state, os.path.join(ckpt_dir, "latest.pt"))
            torch.save(ckpt_state, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))

    wandb.finish()
    print("Training Complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 实验配置
    parser.add_argument("--run_name", type=str, default="Kuramoto_Interpolant_SOCOFing")
    parser.add_argument("--wandb_project", type=str, default="Orientation_Diffusion")
    
    # 数据配置
    parser.add_argument("--snapshot_dir", type=str, default="./data/snapshots/socofing_kura_200_11")
    parser.add_argument("--num_snapshots", type=int, default=11, help="N+1 个快照的数量 (A_0 到 A_N)")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--sigma_max", type=float, default=1.0)
    parser.add_argument("--lambda_E", type=float, default=1.0, help="网络 E (噪声) 的 Loss 权重")
    
    # 运行与记录频率
    parser.add_argument("--eval_freq", type=int, default=10, help="每隔多少 Epoch 在 WandB 上生成一次图片")
    parser.add_argument("--save_freq", type=int, default=200, help="每隔多少 Epoch 保存一次权重")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    train_interpolant(args)
import torch
import torch.nn.functional as F

class DecoupledInterpolantLoss:
    """
    解耦的流匹配与分数匹配联合损失函数。
    彻底抛弃了 Wrapped Gaussian 和蒙特卡洛多重采样。
    """
    def __init__(self, interpolant_engine, lambda_E=1.0, time_scale=1000.0):
        """
        Args:
            interpolant_engine: 我们之前写的数学枢纽 (InterpolantEngine)
            lambda_E: 平衡速度网络 (D) 和噪声网络 (E) Loss 的权重。通常设为 1.0 即可。
            time_scale: 因为我们采样的 t 是 (0, 1) 的连续实数，而原版 U-Net 的 
                        Sinusoidal Positional Encoding 习惯接收大数值 (如 0~1000)。
                        这个缩放因子可以保证网络时间特征提取的正常运作。
        """
        self.engine = interpolant_engine
        self.lambda_E = lambda_E
        self.time_scale = time_scale

    def __call__(self, model_D, model_E, snapshots):
        """
        计算单步的联合 Loss。
        
        Args:
            model_D: 预测物理结构漂移的网络
            model_E: 预测高斯噪声的网络
            snapshots: DataLoader 读出来的无界展开快照，Shape: (B, N, C, H, W)
            
        Returns:
            loss: 总损失
            loss_dict: 包含各自分项的字典，方便记录到 WandB
        """
        B, _, C, H, W = snapshots.shape
        device = snapshots.device

        # 1. 采样时间 t ~ U(0, 1) 和纯高斯噪声 epsilon ~ N(0, I)
        t = torch.rand((B,), device=device)
        epsilon = torch.randn((B, C, H, W), device=device)

        # 2. 从数学引擎中获取终极真理 (Targets) 与当前带噪状态 (x_t)
        # x_t 也是定义在无限实数轴上的展开角度
        x_t, target_D, target_E = self.engine.get_train_targets(snapshots, t, epsilon)

        # 3. 缩放时间 t 喂给网络 (适配原版 U-Net 的周期嵌入)
        t_model = t * self.time_scale

        # 4. 网络前向传播
        # 注意：原版 modules.py 里的 U-Net 已经在内部第一层做了 
        # x = torch.cat([x.cos(), x.sin()], dim=1)
        # 所以我们直接把 x_t 扔进去，网络绝不会发生 2pi 溢出的困惑！
        pred_D = model_D(x_t, t_model)
        pred_E = model_E(x_t, t_model)

        # 5. 暴力极简的 MSE 损失
        # 没有任何概率密度的对数计算，就是最纯粹的向量回归
        loss_D = F.mse_loss(pred_D, target_D)
        loss_E = F.mse_loss(pred_E, target_E)

        # 6. 组合总损失
        total_loss = loss_D + self.lambda_E * loss_E

        # 打包以便于监控
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_D_drift": loss_D.item(),
            "loss_E_noise": loss_E.item(),
        }

        return total_loss, loss_dict
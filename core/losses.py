import torch
import torch.nn.functional as F
from core.epsilon_provider import build_epsilon_provider


class DecoupledInterpolantLoss:
    """
    解耦的流匹配与分数匹配联合损失函数。
    彻底抛弃了 Wrapped Gaussian 和蒙特卡洛多重采样。
    """
    def __init__(
        self,
        interpolant_engine,
        lambda_E=1.0,
        time_scale=1000.0,
        epsilon_mode="fresh",
        epsilon_seed=12345,
        time_mode="continuous",
        num_time_bins=16,
    ):
        self.engine = interpolant_engine
        self.lambda_E = lambda_E
        self.time_scale = time_scale

        self.epsilon_mode = epsilon_mode
        self.epsilon_provider = build_epsilon_provider(
            mode=epsilon_mode,
            base_seed=epsilon_seed,
        )

        self.time_mode = time_mode
        self.num_time_bins = num_time_bins

    def _sample_t(self, B, device):
        if self.time_mode == "continuous":
            return torch.rand((B,), device=device)

        if self.time_mode == "grid":
            # 采样 bin 中心，避免精确落在 0 或 1
            # t in {(0.5/K), (1.5/K), ..., ((K-0.5)/K)}
            bins = torch.randint(0, self.num_time_bins, (B,), device=device)
            t = (bins.float() + 0.5) / self.num_time_bins
            return t

        raise ValueError(f"不支持的 time_mode: {self.time_mode}")

    def __call__(self, model_D, model_E, snapshots, sample_ids=None):
        """
        Args:
            model_D: 预测物理结构漂移的网络
            model_E: 预测高斯噪声的网络
            snapshots: Shape (B, N, C, H, W)
            sample_ids: 当 epsilon_mode=fixed_per_sample 时必须提供
        """
        B, _, C, H, W = snapshots.shape
        device = snapshots.device
        dtype = snapshots.dtype

        # 1. 采样时间 t
        t = self._sample_t(B, device)

        # 2. epsilon 由 provider 决定
        epsilon = self.epsilon_provider(
            sample_ids=sample_ids,
            shape=(B, C, H, W),
            device=device,
            dtype=dtype,
        )

        # 3. 构造 x_t 与 targets
        x_t, target_D, target_E = self.engine.get_train_targets(snapshots, t, epsilon)

        # 4. 网络前向
        t_model = t * self.time_scale
        pred_D = model_D(x_t, t_model)
        pred_E = model_E(x_t, t_model)

        # 5. MSE loss
        loss_D = F.mse_loss(pred_D, target_D)
        loss_E = F.mse_loss(pred_E, target_E)

        total_loss = loss_D + self.lambda_E * loss_E

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_D_drift": loss_D.item(),
            "loss_E_noise": loss_E.item(),
        }

        return total_loss, loss_dict
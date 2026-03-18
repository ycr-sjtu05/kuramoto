import torch
import math

class BaseInterpolator:
    """
    插值引擎基类。
    如果你以后想换成纯线性插值 (Linear OT) 或者球面插值 (SLERP)，
    只需要继承这个类并实现 compute_mu_and_dot 即可。
    """
    def compute_mu_and_dot(self, snapshots, t):
        raise NotImplementedError


class BernsteinInterpolator(BaseInterpolator):
    """
    基于伯恩斯坦多项式 (Bernstein Polynomials) 的 N 阶平滑插值引擎。
    """
    def __init__(self, num_snapshots, device="cuda"):
        # N 个快照对应 N-1 阶多项式
        self.N = num_snapshots - 1  
        self.device = device
        
        # 预计算二项式系数 (Binomial Coefficients) 以极限提升训练时的速度
        # C(N, i)
        self.binom_N = torch.tensor(
            [math.comb(self.N, i) for i in range(self.N + 1)], 
            device=device, dtype=torch.float32
        )
        # C(N-1, i)，用于计算导数
        if self.N > 0:
            self.binom_N_minus_1 = torch.tensor(
                [math.comb(self.N - 1, i) for i in range(self.N)], 
                device=device, dtype=torch.float32
            )
        else:
            self.binom_N_minus_1 = torch.tensor([1.0], device=device, dtype=torch.float32)

    def compute_mu_and_dot(self, snapshots, t):
        # 【防御性编程 A】以输入张量的设备为唯一真理
        device = snapshots.device 
        t = t.to(device)
        self.binom_N = self.binom_N.to(device)
        self.binom_N_minus_1 = self.binom_N_minus_1.to(device)

        B, N_plus_1, C, H, W = snapshots.shape
        t_exp = t.view(-1, 1).clamp(1e-7, 1 - 1e-7)

        i = torch.arange(self.N + 1, device=device).float().view(1, -1)
        term1 = t_exp ** i
        term2 = (1 - t_exp) ** (self.N - i)
        B_t = self.binom_N.view(1, -1) * term1 * term2

        mu_t = torch.einsum('bn,bnchw->bchw', B_t, snapshots)

        if self.N == 0:
            mu_dot_t = torch.zeros_like(mu_t)
        else:
            j = torch.arange(self.N, device=device).float().view(1, -1)
            B_minus_1 = self.binom_N_minus_1.view(1, -1) * (t_exp ** j) * ((1 - t_exp) ** (self.N - 1 - j))
            
            dot_B_t = torch.zeros((B, self.N + 1), device=device)
            dot_B_t[:, 0] = -self.N * B_minus_1[:, 0]
            dot_B_t[:, self.N] = self.N * B_minus_1[:, self.N - 1]
            if self.N > 1:
                dot_B_t[:, 1:self.N] = self.N * (B_minus_1[:, :-1] - B_minus_1[:, 1:])

            mu_dot_t = torch.einsum('bn,bnchw->bchw', dot_B_t, snapshots)

        return mu_t, mu_dot_t


class NoiseSchedule:
    def __init__(self, sigma_max=1.0, eps=1e-5):
        self.sigma_max = sigma_max
        self.eps = eps 

    def compute_sigma_and_dot(self, t):
        t_exp = t.view(-1, 1, 1, 1)
        
        # 1. sigma_mix: 纯粹的数学调度，严格保证 t=0 时 sigma=0
        sigma_mix = self.sigma_max * (1 - torch.cos(torch.pi / 2 * t_exp))
        
        # 2. sigma_safe: 用于公式里的分母，垫了一层 eps 绝对防止除 0
        sigma_safe = sigma_mix + self.eps
        
        dot_sigma_t = self.sigma_max * (torch.pi / 2) * torch.sin(torch.pi / 2 * t_exp)
        
        return sigma_mix, sigma_safe, dot_sigma_t


class InterpolantEngine:
    """
    前向与后向总枢纽。
    隔离所有的数学脏活，为 train.py 和 sample.py 提供最干净的 API。
    """
    def __init__(self, num_snapshots, sigma_max=1.0, eps=1e-5, device="cuda"):
        self.math_engine = BernsteinInterpolator(num_snapshots, device=device)
        self.noise_schedule = NoiseSchedule(sigma_max=sigma_max, eps=eps)

    

    def get_train_targets(self, snapshots, t, epsilon):
        mu_t, mu_dot_t = self.math_engine.compute_mu_and_dot(snapshots, t)
        
        # 拿取不含 eps 的纯净 sigma_mix
        sigma_mix, _, _ = self.noise_schedule.compute_sigma_and_dot(t)

        # 完美对齐理论：t=0 时，x_0 绝对纯净
        x_t = mu_t + sigma_mix * epsilon 
        
        target_D = mu_dot_t
        target_E = epsilon

        return x_t, target_D, target_E


    def get_reverse_drift(self, x_t, t, pred_D, pred_E, g_t):
        # 推理时，分母使用带有 eps 的 sigma_safe 防御 NaN
        _, sigma_safe, dot_sigma_t = self.noise_schedule.compute_sigma_and_dot(t)
        
        # 【防御性编程 C】强制对齐设备和数据类型
        if not isinstance(g_t, torch.Tensor):
            g_t = torch.tensor(g_t, device=x_t.device, dtype=x_t.dtype)
        
        v_flow = pred_D + dot_sigma_t * pred_E
        
        # 除以 sigma_safe，稳如磐石
        score_correction = 0.5 * (g_t ** 2) * (pred_E / sigma_safe) 
        
        reverse_drift = v_flow + score_correction
        return reverse_drift
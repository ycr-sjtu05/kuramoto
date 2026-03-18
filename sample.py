import torch
import torch.nn as nn
import numpy as np
import math
import os
from tqdm.auto import tqdm
from torchvision.utils import save_image

def phase_modulate(x):
    """将实数轴上的角度强行折叠回 [-pi, pi] 主值区间"""
    return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

class DualHeadModel(nn.Module):
    """
    一个极其轻量的包装器。
    作用是骗过 fid.py 里的 `self.model.eval()`，让 D 和 E 两个网络作为一个整体被调用。
    """
    def __init__(self, model_D, model_E):
        super().__init__()
        self.D = model_D
        self.E = model_E

class KuramotoSampler:
    """
    全能的推理采样器。
    支持纯 ODE 极速采样，也支持带控制变量 g(t) 的 SDE 采样。
    """
    def __init__(self, engine, img_size=96, channels=1, steps=100, noise_scale=0.0, time_scale=1000.0, device="cuda"):
        """
        Args:
            engine: InterpolantEngine 数学枢纽
            steps: 采样的步数 (比如 100, 50, 甚至 20)
            noise_scale: SDE 随机扰动强度。设为 0.0 就是纯 ODE (最快最平滑)，> 0 就是 SDE。
            time_scale: 与训练时保持一致的频率缩放因子 (1000.0)
        """
        self.engine = engine
        self.img_size = img_size
        self.channels = channels
        self.steps = steps
        self.noise_scale = noise_scale 
        self.time_scale = time_scale
        self.device = device
        
        # 为了兼容 fid.py，预存一下 noise_steps 属性 (fid.py 可能不用，但备着安全)
        self.noise_steps = steps 

    def sample(self, dual_model, n, run_name=None):
        """
        从纯噪声生成图像。
        fid.py 中的 fid_score_noise 会调用这个接口。
        """
        dual_model.eval()
        
        # 1. 初始化先验状态 (Prior)
        # 因为我们是在实数轴展开的，先验就是 N(0, sigma_max^2 * I)
        sigma_max = self.engine.noise_schedule.sigma_max
        x = torch.randn(n, self.channels, self.img_size, self.img_size, device=self.device) * sigma_max
        x = phase_modulate(x)

        dt = 1.0 / self.steps

        with torch.no_grad():
            # 时间 t 从 1.0 倒退回 0.0 (实际上取到 > 0 的极小值)
            for i in tqdm(reversed(range(1, self.steps + 1)), desc="Sampling ODE/SDE", leave=False):
                # 当前的真实物理时间 t
                t_val = i / self.steps
                t = torch.full((n,), t_val, device=self.device)
                
                # 适配网络的放大时间
                t_model = t * self.time_scale
                
                # 网络预测
                pred_D = dual_model.D(x, t_model)
                pred_E = dual_model.E(x, t_model)
                
                sigma_mix, _, _ = self.engine.noise_schedule.compute_sigma_and_dot(t)
                
                # SDE 的随机扰动严格挂钩纯净的 sigma_mix，保证 t=0 时绝对干净
                g_t = self.noise_scale * sigma_mix.view(-1, 1, 1, 1)
                
                # 调用数学枢纽获取真正的物理更新速度 (Drift)
                drift = self.engine.get_reverse_drift(x, t, pred_D, pred_E, g_t)
                
                # 欧拉更新: x_{old} - Drift * dt + g(t) * sqrt(dt) * Z
                z = torch.randn_like(x) if self.noise_scale > 0 else 0.0
                x = x - drift * dt + g_t * math.sqrt(dt) * z
                
                # 跑完一步，强行拉回 [-pi, pi] 主值区间
                x = phase_modulate(x)

        dual_model.train()
        return x

    def sample_image(self, dual_model, x_real):
        """
        重构/加噪去噪生成。
        fid.py 中的 fid_score_image 会调用这个接口。
        """
        dual_model.eval()
        B = x_real.shape[0]
        
        # 为了兼容 FID 评测，我们需要先将真实图片“破坏”到 t=1 的状态，然后再生成回来
        t_1 = torch.ones((B,), device=self.device)
        sigma_max, _, _ = self.engine.noise_schedule.compute_sigma_and_dot(t_1)
        
        # 直接利用公式跳跃到先验 (假设此时锚点 mu_1 趋近于 0，纯靠高斯噪声主导)
        epsilon = torch.randn_like(x_real)
        x_noisy = x_real * 0.0 + sigma_max.view(-1, 1, 1, 1) * epsilon
        x = phase_modulate(x_noisy)
        
        # 剩下的逻辑与 sample() 完全一致，执行反向过程
        dt = 1.0 / self.steps
        with torch.no_grad():
            for i in reversed(range(1, self.steps + 1)):
                t_val = i / self.steps
                t = torch.full((B,), t_val, device=self.device)
                t_model = t * self.time_scale
                
                pred_D = dual_model.D(x, t_model)
                pred_E = dual_model.E(x, t_model)
                
                sigma_mix, _, _ = self.engine.noise_schedule.compute_sigma_and_dot(t)
                g_t = self.noise_scale * sigma_mix.view(-1, 1, 1, 1)
                
                drift = self.engine.get_reverse_drift(x, t, pred_D, pred_E, g_t)
                z = torch.randn_like(x) if self.noise_scale > 0 else 0.0
                x = x - drift * dt + g_t * math.sqrt(dt) * z
                x = phase_modulate(x)

        dual_model.train()
        return x
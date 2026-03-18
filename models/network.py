import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 基础组件：直接继承自原版代码，一字不改，保证感受野和特征提取能力对齐
# ==========================================
class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())




import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, size, num_heads=4):
        """
        引入多头机制 (Multi-Head Self-Attention)
        :param channels: 输入的通道数
        :param size: 输入图像的高/宽 (对于注意力重塑很重要)
        :param num_heads: 注意力头的数量 (必须能被 channels 整除)
        """
        super().__init__()
        self.channels = channels
        self.size = size
        self.num_heads = num_heads
        
        # 确保通道数可以被头数平分
        assert channels % num_heads == 0, f"通道数 {channels} 必须能被头数 {num_heads} 整除"
        self.head_dim = channels // num_heads
        
        # 一次性生成 Q, K, V，比分三次算更省显存和时间
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 多头融合后的输出映射
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 组归一化 (GroupNorm) 配合 Attention 效果比 BatchNorm 更好
        self.norm = nn.GroupNorm(num_groups=32 if channels >= 32 else 1, num_channels=channels)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W  # 序列长度
        
        # 1. 归一化 (Pre-Norm 结构，训练更稳)
        x_norm = self.norm(x)
        
        # 2. 生成 Q, K, V 并切分
        qkv = self.qkv(x_norm)  # (B, 3*C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # 每个都是 (B, C, H, W)
        
        # 3. 重塑为多头形状: (Batch, Heads, Head_dim, Sequence_length)
        q = q.view(B, self.num_heads, self.head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)
        
        # 4. 计算注意力权重: Q * K^T / sqrt(d)
        # 用 einsum 极其优雅地处理矩阵乘法
        # q: (B, Heads, Head_dim, N)
        # k: (B, Heads, Head_dim, N)
        # attn: (B, Heads, N, N) -> NxN 的注意力矩阵
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # 5. 聚合 Value: attn * V
        # v: (B, Heads, Head_dim, N) (注意这里的 m 对应 Sequence_length)
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # 6. 把多头拼接回原本的形状 (B, C, H, W)
        out = out.reshape(B, C, H, W)
        
        # 7. 线性映射并加上残差连接 (Residual Connection)
        out = self.proj(out)
        return x + out



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

# ==========================================
# 核心架构：为 Interpolant 框架专属打造的 PhaseUNet
# ==========================================
class PhaseUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=512, img_size=96, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        
        # 【架构修改 1】输入通道翻倍 (c_in * 2)，因为我们要容纳 cos 和 sin 特征
        self.inc = DoubleConv(c_in * 2, 64)
        
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128, img_size // 2, num_heads=4)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256, img_size // 4, num_heads=8)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention(256, img_size // 8, num_heads=8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128, img_size // 4, num_heads=8)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64, img_size // 2, num_heads=4)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64, img_size, num_heads=4)
        
        # 【架构修改 2】输出通道直接设为 c_out (而不是 c_out * 2)
        # 因为我们直接回归实数轴上的一维切向量速度或高斯噪声
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        # 抛弃 self.device，以张量 t 当前所在的 device 为绝对准则
        device = t.device
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # 【架构修改 3】天然的防溢出结界
        # 接收的 x 可能是实数轴上任意大的展开角度，直接用三角函数折叠回单位圆
        x = torch.cat([x.cos(), x.sin()], dim=1)
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        
        # 【架构修改 4】直接输出！删掉原版的 angle_space 函数！
        output = self.outc(x)
        return output
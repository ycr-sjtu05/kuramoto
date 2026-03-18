import math
import os
import numpy as np
import torch
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

from utils import map_to_image

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class FIDEvaluation:
    def __init__(
        self,
        batch_size,
        dl,                 # 原始图像 dataloader
        sampler,
        channels=1,
        model=None,         # DualHeadModel(ema_model_D, ema_model_E)
        accelerator=None,
        stats_dir="./fid_stats",
        device="cuda",
        num_fid_samples=50000,
        inception_block_idx=2048,
        dataset="fp",
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.model = model
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print

        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)

        self.dataset_stats_loaded = False
        self.dataset = dataset

    def calculate_inception_features(self, samples):
        samples = samples.to(self.device).float()

        # 【关键修正】：将 [-1, 1] 域平移缩放至 [0, 1] 域，迎合 InceptionV3 的口味
        samples = (samples + 1.0) / 2.0
        # 截断以防万一浮点精度溢出
        samples = samples.clamp(0.0, 1.0)

        if self.channels == 1:
            samples = repeat(samples, "b 1 h w -> b c h w", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))

        features = rearrange(features, "b c 1 1 -> b c")
        return features

    def load_or_precalc_dataset_stats(self):
        os.makedirs(self.stats_dir, exist_ok=True)
        path = os.path.join(self.stats_dir, f"{self.dataset}_stats.npz")

        if os.path.exists(path):
            ckpt = np.load(path)
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            ckpt.close()
            self.print_fn("Dataset stats loaded from disk.")
            self.dataset_stats_loaded = True
            return

        num_batches = int(math.ceil(self.n_samples / self.batch_size))
        stacked_real_features = []

        self.print_fn(f"Stacking Inception features for {self.n_samples} real samples.")
        for i, (images, _) in enumerate(tqdm(self.dl)):
            if i >= num_batches:
                break
            real_samples = images.to(self.device).float()
            real_features = self.calculate_inception_features(real_samples)
            stacked_real_features.append(real_features)

        stacked_real_features = torch.cat(stacked_real_features, dim=0)[:self.n_samples].cpu().numpy()
        self.m2 = np.mean(stacked_real_features, axis=0)
        self.s2 = np.cov(stacked_real_features, rowvar=False)

        np.savez_compressed(path, m2=self.m2, s2=self.s2)
        self.print_fn(f"Dataset stats cached to {path}.")
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score_noise(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()

        self.model.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []

        self.print_fn(f"Stacking Inception features for {self.n_samples} generated samples.")
        for batch in tqdm(batches):
            fake_phases = self.sampler.sample(self.model, batch)
            fake_images = map_to_image(fake_phases)
            fake_features = self.calculate_inception_features(fake_images)
            stacked_fake_features.append(fake_features)

        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        fid = calculate_frechet_distance(m1, s1, self.m2, self.s2)
        self.print_fn(f"FID: {fid:.4f}")
        return fid



if __name__ == "__main__":
    import argparse
    from utils import get_data
    from sample import KuramotoSampler, DualHeadModel
    from models.network import PhaseUNet 
    from core.interpolant_lib import InterpolantEngine

    parser = argparse.ArgumentParser()
    # 1. 数据集配置 (必须和 train.py 保持绝对一致)
    parser.add_argument("--dataset_name", type=str, default="cifar")
    parser.add_argument("--dataset_path", type=str, default="./data/raw")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    
    # 2. 评测超参数配置
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ckpt_path", type=str, required=True, help="训练好的 latest.pt 权重路径")
    parser.add_argument("--num_fid_samples", type=int, default=10000, help="测 FID 生成的图片数量 (快速测试填10000，发论文填50000)")
    parser.add_argument("--sampler_steps", type=int, default=50, help="ODE/SDE 的反向采样步数")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="0.0 为纯 ODE 采样，>0 为 SDE 采样")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # ==========================================
    # 评测流水线组装
    # ==========================================
    
    # 1. 获取真实的 DataLoader (计算 Ground Truth 的特征分布)
    print(f"Loading real dataset [{args.dataset_name}] for ground truth stats...")
    dl, _ = get_data(args)

    # 2. 实例化网络架构 (⚠️ 严格对齐 time_dim=512)
    print("Initializing EMA networks...")
    ema_model_D = PhaseUNet(
        c_in=args.channels, 
        c_out=args.channels, 
        time_dim=512,  # 与你的 network.py 保持一致
        img_size=args.image_size, 
        device=args.device
    ).to(args.device)
    
    ema_model_E = PhaseUNet(
        c_in=args.channels, 
        c_out=args.channels, 
        time_dim=512,  # 与你的 network.py 保持一致
        img_size=args.image_size, 
        device=args.device
    ).to(args.device)

    # 3. 剥离并加载 EMA 影子权重
    print(f"Loading weights from {args.ckpt_path}...")
    ckpt = torch.load(args.ckpt_path, map_location=args.device)
    ema_model_D.load_state_dict(ckpt['ema_model_D'])
    ema_model_E.load_state_dict(ckpt['ema_model_E'])
    
    # 组装为推理专用的双头模型并开启 eval 模式
    dual_model = DualHeadModel(ema_model_D, ema_model_E)
    dual_model.eval()

    # 4. 实例化底层数学引擎与采样器
    engine = InterpolantEngine(num_snapshots=11, sigma_max=1.0, device=args.device)
    sampler = KuramotoSampler(
        engine, 
        img_size=args.image_size, 
        channels=args.channels, 
        steps=args.sampler_steps, 
        noise_scale=args.noise_scale, 
        device=args.device
    )

    # 5. 启动终极 FID 评测
    print(f"\n🚀 Starting FID Evaluation ({args.num_fid_samples} samples) on device: {args.device}...")
    fid_eval = FIDEvaluation(
        batch_size=args.batch_size,
        dl=dl,
        sampler=sampler,
        channels=args.channels,
        model=dual_model,
        device=args.device,
        num_fid_samples=args.num_fid_samples,
        dataset=args.dataset_name
    )
    
    # 从纯高斯先验开始，全量生成新图并计算 FID
    final_fid = fid_eval.fid_score_noise()
    print(f"\n✅ Final FID Score: {final_fid:.4f}")
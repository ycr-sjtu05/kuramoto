import os
import sys
import math
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

# 让脚本能 import 项目根目录模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.interpolant_lib import InterpolantEngine
from models.network import PhaseUNet
from sample import KuramotoSampler, DualHeadModel
from utils import map_to_image


@torch.no_grad()
def generate_samples(args):
    device = args.device

    ckpt = torch.load(args.ckpt_path, map_location=device)

    engine = InterpolantEngine(
        num_snapshots=args.num_snapshots,
        sigma_max=args.sigma_max,
        device=device,
    )

    model_D = PhaseUNet(
        c_in=args.channels,
        c_out=args.channels,
        img_size=args.image_size,
        device=device,
    ).to(device)

    model_E = PhaseUNet(
        c_in=args.channels,
        c_out=args.channels,
        img_size=args.image_size,
        device=device,
    ).to(device)

    # 优先加载 EMA 权重
    if "ema_model_D" in ckpt and "ema_model_E" in ckpt:
        model_D.load_state_dict(ckpt["ema_model_D"])
        model_E.load_state_dict(ckpt["ema_model_E"])
        print("Loaded EMA weights from checkpoint.")
    else:
        model_D.load_state_dict(ckpt["model_D"])
        model_E.load_state_dict(ckpt["model_E"])
        print("EMA weights not found. Loaded raw model weights.")

    model_D.eval()
    model_E.eval()

    dual_model = DualHeadModel(model_D, model_E).to(device).eval()

    sampler = KuramotoSampler(
        engine=engine,
        img_size=args.image_size,
        channels=args.channels,
        steps=args.sampler_steps,
        noise_scale=args.noise_scale,
        device=device,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_imgs = []
    num_done = 0

    pbar = tqdm(total=args.num_samples, desc="Generating samples for NN analysis")
    while num_done < args.num_samples:
        cur_bs = min(args.batch_size, args.num_samples - num_done)
        sampled_phases = sampler.sample(dual_model, n=cur_bs)
        sampled_imgs = map_to_image(sampled_phases).cpu()  # [-1, 1]
        all_imgs.append(sampled_imgs)
        num_done += cur_bs
        pbar.update(cur_bs)
    pbar.close()

    all_imgs = torch.cat(all_imgs, dim=0)
    torch.save(
        {
            "images": all_imgs,  # shape: [N, C, H, W], range [-1, 1]
            "ckpt_path": args.ckpt_path,
            "num_samples": args.num_samples,
            "image_size": args.image_size,
            "channels": args.channels,
            "sampler_steps": args.sampler_steps,
            "noise_scale": args.noise_scale,
        },
        out_path,
    )

    print(f"Saved generated samples to: {out_path}")
    print(f"Tensor shape: {tuple(all_imgs.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--num_snapshots", type=int, default=11)
    parser.add_argument("--sigma_max", type=float, default=1.0)

    parser.add_argument("--sampler_steps", type=int, default=100)
    parser.add_argument("--noise_scale", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    generate_samples(args)
import os
import sys
import json
import csv
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# 让脚本能 import 项目根目录模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from pytorch_fid.inception import InceptionV3


def to_01(x):
    # x in [-1, 1] -> [0, 1]
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def get_cifar_train_test(dataset_path, image_size):
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    train_set = CIFAR10(root=dataset_path, train=True, download=True, transform=tfm)
    test_set = CIFAR10(root=dataset_path, train=False, download=True, transform=tfm)

    train_imgs = torch.stack([train_set[i][0] for i in range(len(train_set))], dim=0)
    test_imgs = torch.stack([test_set[i][0] for i in range(len(test_set))], dim=0)

    return train_imgs, test_imgs


@torch.no_grad()
def extract_inception_features(images, device="cuda", batch_size=128, dims=2048):
    """
    images: [N, C, H, W] in [-1, 1]
    return: [N, dims] float32 on CPU
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    ds = TensorDataset(images)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    feats = []
    for (x,) in tqdm(dl, desc="Extracting Inception features", leave=False):
        x = to_01(x).to(device, non_blocking=True)
        pred = model(x)[0]

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().float()
        feats.append(pred)

    return torch.cat(feats, dim=0)


@torch.no_grad()
def batched_nn(query_feats, ref_feats, device="cuda", query_chunk=256, ref_chunk=4096):
    """
    返回 query 每个样本到 ref 的最近邻距离和索引
    使用平方欧氏距离
    """
    Nq = query_feats.shape[0]
    Nr = ref_feats.shape[0]

    nn_dists = torch.empty(Nq, dtype=torch.float32)
    nn_idxs = torch.empty(Nq, dtype=torch.long)

    ref_norm_all = (ref_feats ** 2).sum(dim=1)

    for qs in tqdm(range(0, Nq, query_chunk), desc="NN search", leave=False):
        qe = min(qs + query_chunk, Nq)
        q = query_feats[qs:qe].to(device)
        q_norm = (q ** 2).sum(dim=1, keepdim=True)

        best_dist = torch.full((qe - qs,), float("inf"), device=device)
        best_idx = torch.full((qe - qs,), -1, dtype=torch.long, device=device)

        for rs in range(0, Nr, ref_chunk):
            re = min(rs + ref_chunk, Nr)
            r = ref_feats[rs:re].to(device)
            r_norm = ref_norm_all[rs:re].to(device)

            # ||q-r||^2 = ||q||^2 + ||r||^2 - 2 q r^T
            dist = q_norm + r_norm.unsqueeze(0) - 2.0 * (q @ r.t())
            dist = torch.clamp(dist, min=0.0)

            cur_best_dist, cur_best_local_idx = dist.min(dim=1)
            update_mask = cur_best_dist < best_dist

            best_dist[update_mask] = cur_best_dist[update_mask]
            best_idx[update_mask] = (cur_best_local_idx[update_mask] + rs)

        nn_dists[qs:qe] = best_dist.sqrt().cpu()
        nn_idxs[qs:qe] = best_idx.cpu()

    return nn_dists, nn_idxs


@torch.no_grad()
def train_self_nn_threshold(train_feats, subset_size=10000, quantile=0.01, device="cuda", chunk=2048):
    """
    用训练集内部的 leave-one-out NN 距离估计一个 copy threshold
    为了省时间，默认只取前 subset_size 个训练样本
    """
    M = min(subset_size, train_feats.shape[0])
    feats = train_feats[:M].to(device)
    norms = (feats ** 2).sum(dim=1)

    all_best = torch.full((M,), float("inf"), device=device)

    for qs in tqdm(range(0, M, chunk), desc="Self-NN threshold", leave=False):
        qe = min(qs + chunk, M)
        q = feats[qs:qe]
        q_norm = norms[qs:qe].unsqueeze(1)

        best = torch.full((qe - qs,), float("inf"), device=device)

        for rs in range(0, M, chunk):
            re = min(rs + chunk, M)
            r = feats[rs:re]
            r_norm = norms[rs:re]

            dist = q_norm + r_norm.unsqueeze(0) - 2.0 * (q @ r.t())
            dist = torch.clamp(dist, min=0.0)

            # 排除自己
            if qs == rs:
                idx = torch.arange(qe - qs, device=device)
                dist[idx, idx] = float("inf")

            cur_best, _ = dist.min(dim=1)
            best = torch.minimum(best, cur_best)

        all_best[qs:qe] = best

    all_best = all_best.sqrt().cpu()
    tau = torch.quantile(all_best, quantile).item()
    return tau, all_best


def save_histogram(values_dict, out_path, title, xlabel):
    plt.figure(figsize=(8, 5))
    for label, values in values_dict.items():
        plt.hist(values.numpy(), bins=60, alpha=0.45, density=True, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_topk_visualization(gen_imgs, train_imgs, test_imgs,
                            train_nn_idx, test_nn_idx, train_nn_dist, test_nn_dist,
                            label, out_path, topk=20):
    order = torch.argsort(train_nn_dist)[:topk]

    fig, axes = plt.subplots(topk, 3, figsize=(9, 3 * topk))
    if topk == 1:
        axes = axes[None, :]

    for row, idx in enumerate(order.tolist()):
        g = to_01(gen_imgs[idx]).permute(1, 2, 0).numpy()
        tr = to_01(train_imgs[train_nn_idx[idx]]).permute(1, 2, 0).numpy()
        te = to_01(test_imgs[test_nn_idx[idx]]).permute(1, 2, 0).numpy()

        axes[row, 0].imshow(g)
        axes[row, 0].set_title(f"{label} gen\nidx={idx}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(tr)
        axes[row, 1].set_title(f"train NN\nL2={train_nn_dist[idx]:.4f}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(te)
        axes[row, 2].set_title(f"test NN\nL2={test_nn_dist[idx]:.4f}")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_name != "cifar":
        raise NotImplementedError("当前脚本先只支持 CIFAR。")

    if len(args.gen_paths) != len(args.labels):
        raise ValueError("--gen_paths 和 --labels 的数量必须一致。")

    print("Loading CIFAR train/test images...")
    train_imgs, test_imgs = get_cifar_train_test(args.dataset_path, args.image_size)

    print("Extracting Inception features for real train/test...")
    train_feats = extract_inception_features(
        train_imgs, device=args.device, batch_size=args.feat_batch_size, dims=args.fid_dims
    )
    test_feats = extract_inception_features(
        test_imgs, device=args.device, batch_size=args.feat_batch_size, dims=args.fid_dims
    )

    print("Estimating copy threshold from train self-NN...")
    copy_tau, self_nn_dists = train_self_nn_threshold(
        train_feats,
        subset_size=args.self_nn_subset,
        quantile=args.copy_quantile,
        device=args.device,
        chunk=args.self_nn_chunk,
    )

    summary = {
        "dataset_name": args.dataset_name,
        "image_size": args.image_size,
        "fid_dims": args.fid_dims,
        "copy_threshold_quantile": args.copy_quantile,
        "copy_threshold_value": copy_tau,
    }

    hist_train = {}
    hist_delta = {}

    per_sample_csv = out_dir / "per_sample_metrics.csv"
    with open(per_sample_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "label", "sample_idx",
            "train_nn_dist", "test_nn_dist", "delta_test_minus_train",
            "train_nn_idx", "test_nn_idx", "is_copy_suspect"
        ])

        for gen_path, label in zip(args.gen_paths, args.labels):
            print(f"\n=== Processing {label} ===")
            payload = torch.load(gen_path, map_location="cpu")
            gen_imgs = payload["images"].float()

            print("Extracting Inception features for generated images...")
            gen_feats = extract_inception_features(
                gen_imgs, device=args.device, batch_size=args.feat_batch_size, dims=args.fid_dims
            )

            print("Searching nearest neighbors in TRAIN set...")
            train_nn_dist, train_nn_idx = batched_nn(
                gen_feats, train_feats,
                device=args.device,
                query_chunk=args.query_chunk,
                ref_chunk=args.ref_chunk,
            )

            print("Searching nearest neighbors in TEST set...")
            test_nn_dist, test_nn_idx = batched_nn(
                gen_feats, test_feats,
                device=args.device,
                query_chunk=args.query_chunk,
                ref_chunk=args.ref_chunk,
            )

            delta = test_nn_dist - train_nn_dist
            is_copy = train_nn_dist < copy_tau

            hist_train[label] = train_nn_dist
            hist_delta[label] = delta

            summary[label] = {
                "num_gen": int(gen_imgs.shape[0]),
                "train_nn_mean": float(train_nn_dist.mean().item()),
                "train_nn_median": float(train_nn_dist.median().item()),
                "test_nn_mean": float(test_nn_dist.mean().item()),
                "test_nn_median": float(test_nn_dist.median().item()),
                "delta_mean": float(delta.mean().item()),
                "delta_median": float(delta.median().item()),
                "copy_rate": float(is_copy.float().mean().item()),
                "num_copy_suspects": int(is_copy.sum().item()),
            }

            for i in range(gen_imgs.shape[0]):
                writer.writerow([
                    label, i,
                    float(train_nn_dist[i].item()),
                    float(test_nn_dist[i].item()),
                    float(delta[i].item()),
                    int(train_nn_idx[i].item()),
                    int(test_nn_idx[i].item()),
                    int(is_copy[i].item()),
                ])

            save_topk_visualization(
                gen_imgs=gen_imgs,
                train_imgs=train_imgs,
                test_imgs=test_imgs,
                train_nn_idx=train_nn_idx,
                test_nn_idx=test_nn_idx,
                train_nn_dist=train_nn_dist,
                test_nn_dist=test_nn_dist,
                label=label,
                out_path=out_dir / f"topk_{label}.png",
                topk=args.topk_vis,
            )

    save_histogram(
        hist_train,
        out_dir / "hist_train_nn.png",
        title="Generated -> Train NN Distance",
        xlabel="Feature-space NN distance",
    )
    save_histogram(
        hist_delta,
        out_dir / "hist_delta.png",
        title="Delta = d_test - d_train",
        xlabel="Feature-space distance gap",
    )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved analysis results to: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default="cifar")
    parser.add_argument("--dataset_path", type=str, default="./data")
    parser.add_argument("--image_size", type=int, default=32)

    parser.add_argument("--gen_paths", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--fid_dims", type=int, default=2048)
    parser.add_argument("--feat_batch_size", type=int, default=128)
    parser.add_argument("--query_chunk", type=int, default=256)
    parser.add_argument("--ref_chunk", type=int, default=4096)

    parser.add_argument("--self_nn_subset", type=int, default=10000)
    parser.add_argument("--self_nn_chunk", type=int, default=2048)
    parser.add_argument("--copy_quantile", type=float, default=0.01)

    parser.add_argument("--topk_vis", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
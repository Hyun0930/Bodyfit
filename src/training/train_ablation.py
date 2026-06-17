"""Ablation 1/3/4 변형 모델 학습 스크립트.

Usage:
    # Ablation 1: 체형 조건화 제거
    python3 -m src.training.train_ablation --variant no_cond ...

    # Ablation 3: ST-GCN → MLP
    python3 -m src.training.train_ablation --variant mlp_feat ...

    # Ablation 4: Cluster body encoding
    python3 -m src.training.train_ablation --variant cluster_cond ...
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data import BodyFitDataset
from src.evaluation.ablation import ClusterCondBCSTNF, MLPFeatureBCSTNF, NoCondBCSTNF

EXERCISES = ["squat", "bench", "deadlift", "ohp"]
VARIANTS = ["no_cond", "mlp_feat", "cluster_cond"]


def get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_body_stats(subset):
    bodies = np.stack([subset[i][1].numpy() for i in range(len(subset))])
    mean = torch.tensor(bodies.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(bodies.std(axis=0).clip(min=1e-6), dtype=torch.float32)
    return mean, std


def compute_cluster_weights(dataset, n_clusters: int = 10) -> torch.Tensor:
    bodies = np.stack([dataset[i][1].numpy() for i in range(len(dataset))])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(bodies)
    labels = km.labels_
    counts = np.bincount(labels, minlength=n_clusters)
    cluster_w = 1.0 / counts.clip(min=1)
    sample_w = cluster_w[labels]
    sample_w = sample_w / sample_w.sum() * len(sample_w)
    return torch.tensor(sample_w, dtype=torch.float32)


def train_epoch(model, loader, optimizer, device, body_mean, body_std):
    model.train()
    total = 0.0
    for pose, body in loader:
        pose, body = pose.to(device), body.to(device)
        body = (body - body_mean) / body_std
        loss = model(pose, body).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def val_epoch(model, loader, device, body_mean, body_std):
    model.eval()
    total = 0.0
    for pose, body in loader:
        pose, body = pose.to(device), body.to(device)
        body = (body - body_mean) / body_std
        total += model(pose, body).mean().item()
    return total / len(loader)


def build_model(variant: str) -> torch.nn.Module:
    if variant == "no_cond":
        return NoCondBCSTNF()
    elif variant == "mlp_feat":
        return MLPFeatureBCSTNF()
    elif variant == "cluster_cond":
        return ClusterCondBCSTNF()
    else:
        raise ValueError(f"Unknown variant: {variant}. Choose from {VARIANTS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--ckpt_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_clusters", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Variant: {args.variant} | Device: {device}")

    ckpt_dir = Path(args.ckpt_dir) / args.variant
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = BodyFitDataset(exercises=EXERCISES, root=Path(args.data_root))
    train_set, val_set = dataset.split(train_ratio=0.9, seed=42)
    print(f"Train: {len(train_set)} | Val: {len(val_set)}")

    body_mean, body_std = compute_body_stats(train_set)
    body_mean, body_std = body_mean.to(device), body_std.to(device)

    # cluster_cond는 학습 전 k-means fit 필요
    model = build_model(args.variant)
    if args.variant == "cluster_cond":
        bodies = np.stack([train_set[i][1].numpy() for i in range(len(train_set))])
        model.body_enc.fit(bodies)
        print(f"ClusterBodyEncoder fitted with {args.n_clusters} clusters")

    model = model.to(device)

    weights = compute_cluster_weights(train_set, n_clusters=args.n_clusters)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, body_mean, body_std)
        vl = val_epoch(model, val_loader, device, body_mean, body_std)
        scheduler.step()
        train_losses.append(tr)
        val_losses.append(vl)
        print(f"Epoch {epoch:3d}/{args.epochs} | train NLL {tr:.4f} | val NLL {vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_nll": vl,
                    "variant": args.variant,
                    "body_mean": body_mean.cpu(),
                    "body_std": body_std.cpu(),
                },
                ckpt_dir / "best.pt",
            )
            print(f"  → saved best (val_nll={best_val:.4f})")

    Path("results").mkdir(exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="train NLL")
    ax.plot(val_losses, label="val NLL")
    ax.set_xlabel("epoch")
    ax.legend()
    fig.savefig(f"results/{args.variant}_train_curve.png", dpi=100)
    plt.close(fig)
    print(f"Done. Checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

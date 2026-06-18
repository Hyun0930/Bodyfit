"""정상/이상 anomaly score 분포 비교 (NLL / z-norm / log_det).

Usage:
    python3 -m src.evaluation.score_dist \
        --ckpt /home/work/body_fit/checkpoints/bc_stnf/best.pt \
        --tier3_root /home/work/body_fit/test \
        --device cuda
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data.tier3_dataset import Tier3Dataset
from src.models.bc_stnf import BCSTNF


def collect(model, score_fn, loader, device, body_mean, body_std):
    normal, abnormal = [], []
    with torch.no_grad():
        for pose, body, label in loader:
            pose, body = pose.to(device), body.to(device)
            if body_mean is not None:
                body = (body - body_mean) / body_std
            scores = score_fn(pose, body).cpu().numpy()
            for s, l in zip(scores, label.numpy()):
                (normal if l == 0 else abnormal).append(s)
    return np.array(normal), np.array(abnormal)


def report(name, normal, abnormal):
    all_scores = np.concatenate([normal, abnormal])
    all_labels = np.array([0] * len(normal) + [1] * len(abnormal))
    auroc = roc_auc_score(all_labels, all_scores)
    print(f"\n=== {name} ===")
    print(f"  정상 ({len(normal):3d}개): mean={normal.mean():.2f}  std={normal.std():.2f}")
    print(f"  이상 ({len(abnormal):3d}개): mean={abnormal.mean():.2f}  std={abnormal.std():.2f}")
    print(f"  정상 mean > 이상 mean (역전): {normal.mean() > abnormal.mean()}")
    print(f"  AUROC: {auroc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--tier3_root", required=True, type=Path)
    parser.add_argument("--labels_path", type=Path, default=None)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = BCSTNF().to(args.device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    body_mean = ckpt.get("body_mean")
    body_std = ckpt.get("body_std")
    if body_mean is not None:
        body_mean = body_mean.to(args.device)
        body_std = body_std.to(args.device)

    labels_path = args.labels_path or args.tier3_root / "labels.json"
    ds = Tier3Dataset(root=args.tier3_root, labels_path=labels_path, max_per_class=args.max_per_class)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    print(f"평가셋: {ds.label_counts()}")

    for name, fn in [
        ("NLL (기존)", model.anomaly_score),
        ("z-norm (||z||²/D)", model.znorm_score),
        ("-log_det", model.logdet_score),
    ]:
        n, ab = collect(model, fn, loader, args.device, body_mean, body_std)
        report(name, n, ab)


if __name__ == "__main__":
    main()

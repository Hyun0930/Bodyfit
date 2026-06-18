"""정상/이상 anomaly score 분포 확인 스크립트.

Usage:
    python3 -m src.evaluation.score_dist \
        --ckpt /home/work/body_fit/checkpoints/bc_stnf/best.pt \
        --tier3_root /home/work/body_fit/test \
        --device cuda
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.tier3_dataset import Tier3Dataset
from src.models.bc_stnf import BCSTNF
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--tier3_root", required=True, type=Path)
    parser.add_argument("--labels_path", type=Path, default=None)
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
    ds = Tier3Dataset(root=args.tier3_root, labels_path=labels_path)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    normal_scores, abnormal_scores = [], []
    with torch.no_grad():
        for pose, body, label in loader:
            pose, body = pose.to(args.device), body.to(args.device)
            if body_mean is not None:
                body = (body - body_mean) / body_std
            scores = model.anomaly_score(pose, body).cpu().numpy()
            for s, l in zip(scores, label.numpy()):
                if l == 0:
                    normal_scores.append(s)
                else:
                    abnormal_scores.append(s)

    normal_scores = np.array(normal_scores)
    abnormal_scores = np.array(abnormal_scores)

    print(f"\n=== Score 분포 (BC-STNF) ===")
    print(f"정상  ({len(normal_scores):3d}개): mean={normal_scores.mean():.2f}  std={normal_scores.std():.2f}  min={normal_scores.min():.2f}  max={normal_scores.max():.2f}")
    print(f"이상  ({len(abnormal_scores):3d}개): mean={abnormal_scores.mean():.2f}  std={abnormal_scores.std():.2f}  min={abnormal_scores.min():.2f}  max={abnormal_scores.max():.2f}")
    print(f"\n정상 mean > 이상 mean: {normal_scores.mean() > abnormal_scores.mean()} (True면 점수 역전 — 이상이 낮은 NLL)")

    # 겹치는 비율
    overlap = np.mean(abnormal_scores < normal_scores.mean())
    print(f"이상 샘플 중 정상 mean보다 낮은 비율: {overlap:.1%}")

    # percentile 확인
    for p in [10, 25, 50, 75, 90]:
        print(f"  정상 {p}th percentile: {np.percentile(normal_scores, p):.2f} | 이상 {p}th percentile: {np.percentile(abnormal_scores, p):.2f}")


if __name__ == "__main__":
    main()

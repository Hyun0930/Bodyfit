"""val set 95th percentile threshold 계산 후 체크포인트에 저장.

Usage:
    cd /home/work/body_fit
    python scripts/calc_threshold.py
    python scripts/calc_threshold.py --data_root data/processed --ckpt checkpoints/cvae/best.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.cvae import CVAE
from src.data import BodyFitDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/processed")
    parser.add_argument("--ckpt", default="checkpoints/cvae/best.pt")
    parser.add_argument("--percentile", type=float, default=95.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.ckpt)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = CVAE().to(device)
    model.load_state_dict(ckpt.get("model_state", ckpt), strict=False)
    model.eval()

    ds = BodyFitDataset(args.data_root, split="val")
    print(f"val set 크기: {len(ds)}")

    scores = []
    with torch.no_grad():
        for i, (pose, body) in enumerate(ds):
            pose = pose.unsqueeze(0).to(device)
            body = body.unsqueeze(0).to(device)
            s = model.anomaly_score(pose, body).item()
            scores.append(s)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{len(ds)} 처리 중...")

    scores = np.array(scores)
    thr = float(np.percentile(scores, args.percentile))

    print(f"\n--- 결과 ---")
    print(f"mean:  {scores.mean():.4f}")
    print(f"std:   {scores.std():.4f}")
    print(f"min:   {scores.min():.4f}")
    print(f"max:   {scores.max():.4f}")
    print(f"threshold ({args.percentile}th percentile): {thr:.4f}")

    ckpt["threshold_95"] = thr
    torch.save(ckpt, ckpt_path)
    print(f"\n체크포인트 저장 완료: {ckpt_path}")


if __name__ == "__main__":
    main()

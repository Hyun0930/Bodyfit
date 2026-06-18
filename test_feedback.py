"""LLM 피드백 파이프라인 동작 테스트 — CVAE 기반.

Usage:
    python3 test_feedback.py
    python3 test_feedback.py --exercise squat --sample_idx 0

ANTHROPIC_API_KEY 필요 (환경변수 또는 /home/work/body_fit/.env)
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

# .env 로드 (서버 환경)
for env_path in [Path("/home/work/body_fit/.env"), Path(".env")]:
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        break

from src.models.cvae import CVAE
from src.evaluation.llm_feedback import feedback_from_model_output

CKPT = Path("/home/work/body_fit/checkpoints/cvae/best.pt")
TEST_ROOT = Path("/home/work/body_fit/test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", default="squat",
                        choices=["squat", "bench", "deadlift", "ohp"])
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY 없음 — .env 파일에 추가하거나 환경변수로 설정하세요.")
        print("  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> /home/work/body_fit/.env")
        return

    device = args.device
    print(f"Device: {device}")

    # CVAE 로드
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = CVAE().to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt, strict=False)
    model.eval()

    body_mean = ckpt.get("body_mean")
    body_std = ckpt.get("body_std")
    if body_mean is not None:
        body_mean = body_mean.to(device)
        body_std = body_std.to(device)

    # 샘플 로드
    ex_dir = TEST_ROOT / args.exercise
    samples = sorted(ex_dir.glob("*.npz"))
    if not samples:
        print(f"샘플 없음: {ex_dir}")
        return
    sample_path = samples[args.sample_idx % len(samples)]

    d = np.load(sample_path, allow_pickle=True)
    pose = torch.tensor(d["pose"], dtype=torch.float32).unsqueeze(0).to(device)
    body = torch.tensor(d["body"], dtype=torch.float32).unsqueeze(0).to(device)
    if body_mean is not None:
        body = (body - body_mean) / body_std

    # 점수 + heatmap
    score = model.anomaly_score(pose, body).item()
    heatmap = model.joint_attribution(pose, body).squeeze(0).cpu().numpy()  # (64, 33)

    # threshold: val 5th percentile 대신 score×0.85 (테스트용)
    threshold = score * 0.85

    print(f"\n샘플: {sample_path.name}")
    print(f"anomaly score : {score:.4f}")
    print(f"threshold     : {threshold:.4f}")
    print(f"heatmap shape : {heatmap.shape}")

    # 피드백 생성
    result = feedback_from_model_output(
        exercise=args.exercise,
        anomaly_score=score,
        threshold=threshold,
        heatmap=heatmap,
        top_k=3,
    )

    print(f"\n=== LLM 피드백 결과 ===")
    print(f"주요 문제 관절 : {result['top_joints']}")
    print(f"이상 판정     : {result['is_anomaly']}")
    print(f"\n피드백:")
    print(result['feedback'])


if __name__ == "__main__":
    main()

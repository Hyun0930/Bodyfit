"""LLM 피드백 파이프라인 동작 테스트."""
import os
import sys
from pathlib import Path
import numpy as np
import torch

# .env 로드
env_path = Path("/home/work/body_fit/.env")
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

from src.models.bc_stnf import BCSTNF
from src.evaluation.llm_feedback import feedback_from_model_output

CKPT = Path("/home/work/body_fit/checkpoints/bc_stnf/best.pt")
# 테스트용 정상 샘플
SAMPLE = next(Path("/home/work/body_fit/test/squat").glob("*.npz"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
model = BCSTNF().to(device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

body_mean = ckpt.get("body_mean")
body_std = ckpt.get("body_std")
if body_mean is not None:
    body_mean = body_mean.to(device)
    body_std = body_std.to(device)

# 샘플 로드
d = np.load(SAMPLE, allow_pickle=True)
pose = torch.tensor(d["pose"], dtype=torch.float32).unsqueeze(0).to(device)
body = torch.tensor(d["body"], dtype=torch.float32).unsqueeze(0).to(device)
if body_mean is not None:
    body = (body - body_mean) / body_std

# anomaly score + heatmap
score = model.anomaly_score(pose, body).item()
heatmap = model.joint_attribution(pose, body).squeeze(0).cpu().numpy()  # (64, 33)

# val set 5th percentile을 threshold로 (단순히 score를 threshold로 사용해서 테스트)
threshold = score * 0.9  # 약간 낮게 설정 → is_anomaly=True 유도

print(f"샘플: {SAMPLE.name}")
print(f"anomaly score: {score:.2f}")
print(f"threshold:     {threshold:.2f}")
print(f"heatmap shape: {heatmap.shape}")
print()

# 피드백 생성
result = feedback_from_model_output(
    exercise="squat",
    anomaly_score=score,
    threshold=threshold,
    heatmap=heatmap,
    top_k=3,
)

print(f"=== LLM 피드백 ===")
print(f"문제 관절: {result['top_joints']}")
print(f"이상 여부: {result['is_anomaly']}")
print(f"피드백:\n{result['feedback']}")

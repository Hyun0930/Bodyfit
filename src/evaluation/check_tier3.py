"""Tier3 npz 파일 데이터 상태 확인."""
import numpy as np
from pathlib import Path

test_dir = Path("/home/work/body_fit/test")

for exercise in ["squat", "bench", "deadlift", "ohp"]:
    ex_dir = test_dir / exercise
    if not ex_dir.exists():
        continue
    files = sorted(ex_dir.glob("*.npz"))[:3]
    print(f"\n=== {exercise} ===")
    for f in files:
        d = np.load(f, allow_pickle=True)
        pose = d["pose"]
        body = d["body"]
        print(f"  {f.name}")
        print(f"    pose: shape={pose.shape} mean={pose.mean():.3f} std={pose.std():.3f} min={pose.min():.3f} max={pose.max():.3f}")
        print(f"    body: {body.round(3)}")
        print(f"    body 범위 이상: {(np.abs(body) > 10).any()}")

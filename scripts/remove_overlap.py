"""Tier 3 평가셋에서 학습 데이터와 겹치는 영상 제거."""
from pathlib import Path

OVERLAP = {
    "bench": ["-J4P-XOEtFY","1mXjWd29xC4","4T9UQ4FBVXI","4Y2ZdHCOXok","5Y3VZsLb1Ys",
              "8d9kbkAuZGI","JvOJdUtx6UQ","QsYre__-aro","SzcSrpVr0GA","ptpmRrzRtWQ",
              "rT7DgCr-3pg","vthMCtgVtFw"],
    "deadlift": ["19ZeTrLZdyQ","CTBiC_tnjOc","VL5Ab0T07e4","XxWcirHIwVo"],
    "ohp": ["4Y2ZdHCOXok","_RlRDWO2jfg"],
    "squat": ["GQ5jj_zH2uA","P-yaD24bUE8"],
}

test_root = Path("/home/work/body_fit/test")
removed = 0
for ex, vids in OVERLAP.items():
    for vid in vids:
        for f in (test_root / ex).glob(f"{vid}_rep*.npz"):
            f.unlink()
            removed += 1
            print(f"  삭제: {f.name}")

print(f"\n완료: {removed}개 rep 삭제")

for ex in ["squat", "bench", "deadlift", "ohp"]:
    n = len(list((test_root / ex).glob("*.npz")))
    print(f"  {ex}: {n} reps 남음")

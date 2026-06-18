"""Synthetic OOD 포즈 생성 — 정상 Tier3 npz에 관절 변형 적용.

전략: 정상 rep의 keypoint 좌표를 의도적으로 벗어나도록 변형
→ 학습 분포에서 명확히 이탈하는 controlled 이상 포즈 생성

변형 종류:
  knee_valgus    — 무릎 내반 (무릎 keypoint를 중앙으로 이동, squat/deadlift)
  forward_lean   — 과도한 앞 기울기 (상체를 z축으로 shift, 전체 운동)
  asymmetry      — 좌우 비대칭 (한 쪽 keypoint에 lateral offset 추가)
  combined       — 위 3가지 동시 적용 (가장 강한 OOD)

Usage:
    python3 -m src.data.synthetic_anomaly \
        --tier3_root /home/work/body_fit/test \
        --output_dir /home/work/body_fit/test_synth \
        --n_per_normal 3
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

# BlazePose 33 keypoint indices
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# 상체 관절 (앞기울기 변형에 사용)
UPPER_BODY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
# 왼쪽 관절
LEFT_SIDE = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
# 오른쪽 관절
RIGHT_SIDE = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

EXERCISES = ["squat", "bench", "deadlift", "ohp"]


def perturb_knee_valgus(pose: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """무릎 내반: 좌우 무릎을 엉덩이 방향으로 당김.

    hip-center 정규화 후 좌표계:
      x: 오른쪽(+), y: 아래(+), z: 깊이(+)
    무릎 내반 = 좌 무릎 x 감소, 우 무릎 x 증가 (중앙으로)
    """
    out = pose.copy()
    # 전 프레임에 걸쳐 적용 (T, 33, 3)
    out[:, LEFT_KNEE, 0] -= strength       # 좌무릎을 오른쪽으로 (중앙으로)
    out[:, RIGHT_KNEE, 0] += strength      # 우무릎을 왼쪽으로 (중앙으로)
    return out


def perturb_forward_lean(pose: np.ndarray, strength: float = 0.4) -> np.ndarray:
    """과도한 앞 기울기: 상체 전체를 z방향으로 shift.

    엉덩이를 축으로 상체가 앞으로 무너지는 시뮬레이션.
    """
    out = pose.copy()
    out[:, UPPER_BODY, 2] += strength
    return out


def perturb_asymmetry(pose: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """좌우 비대칭: 왼쪽 관절에 lateral offset 추가."""
    out = pose.copy()
    out[:, LEFT_SIDE, 0] -= strength       # 왼쪽 전체를 오른쪽으로 (비대칭)
    out[:, LEFT_SIDE, 1] += strength * 0.5  # 약간 위아래 offset도 추가
    return out


def perturb_combined(pose: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """복합 변형: 세 가지 동시 적용."""
    out = perturb_knee_valgus(pose, strength=strength * 1.2)
    out = perturb_forward_lean(out, strength=strength)
    out = perturb_asymmetry(out, strength=strength * 0.8)
    return out


PERTURBATIONS = {
    "knee_valgus": perturb_knee_valgus,
    "forward_lean": perturb_forward_lean,
    "asymmetry": perturb_asymmetry,
    "combined": perturb_combined,
}


def generate_synthetic(
    tier3_root: Path,
    output_dir: Path,
    n_per_normal: int = 3,
    strength: float = 0.3,
    exercises: list = None,
):
    """정상 Tier3 npz → 합성 이상 포즈 생성 + labels.json 작성.

    Returns:
        dict: 생성된 labels (key: exercise/stem, value: {label:1, ...})
    """
    exercises = exercises or EXERCISES
    labels_path = tier3_root / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json 없음: {labels_path}")

    orig_labels = json.loads(labels_path.read_text())
    # 정상(label=0) 샘플만 추출
    normal_keys = [k for k, v in orig_labels.items() if v.get("label") == 0]
    print(f"정상 샘플: {len(normal_keys)}개")

    output_dir.mkdir(parents=True, exist_ok=True)
    synth_labels = {}
    perturbation_cycle = list(PERTURBATIONS.items())

    generated = 0
    for k in normal_keys:
        exercise, stem = k.split("/", 1)
        if exercise not in exercises:
            continue
        src_npz = tier3_root / exercise / f"{stem}.npz"
        if not src_npz.exists():
            continue

        data = np.load(src_npz, allow_pickle=True)
        pose = data["pose"]   # (64, 33, 3)
        body = data["body"]   # (7,)

        ex_out_dir = output_dir / exercise
        ex_out_dir.mkdir(parents=True, exist_ok=True)

        # 사용할 변형 목록 결정 (n_per_normal개)
        perturb_list = [perturbation_cycle[i % len(perturbation_cycle)] for i in range(n_per_normal)]

        for i, (pname, pfn) in enumerate(perturb_list):
            perturbed_pose = pfn(pose, strength=strength)
            out_stem = f"{stem}_synth_{pname}"
            out_path = ex_out_dir / f"{out_stem}.npz"
            np.savez(out_path, pose=perturbed_pose.astype(np.float32), body=body.astype(np.float32))
            synth_labels[f"{exercise}/{out_stem}"] = {
                "label": 1,
                "source": "synthetic",
                "perturbation": pname,
                "strength": strength,
                "original": k,
            }
            generated += 1

    # 원본 정상 샘플도 output_dir에 복사 + labels 포함
    normal_labels = {}
    for k in normal_keys:
        exercise, stem = k.split("/", 1)
        if exercise not in exercises:
            continue
        src_npz = tier3_root / exercise / f"{stem}.npz"
        if not src_npz.exists():
            continue
        dst_dir = output_dir / exercise
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_npz = dst_dir / f"{stem}.npz"
        if not dst_npz.exists():
            shutil.copy2(src_npz, dst_npz)
        normal_labels[k] = {"label": 0, "source": "original"}

    # 통합 labels.json 저장
    combined_labels = {**normal_labels, **synth_labels}
    out_labels = output_dir / "labels.json"
    out_labels.write_text(json.dumps(combined_labels, ensure_ascii=False, indent=2))

    print(f"생성 완료: 정상 {len(normal_labels)}개 + 합성 이상 {generated}개")
    print(f"저장 위치: {output_dir}")
    print(f"labels.json: {out_labels}")
    return combined_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier3_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--n_per_normal", type=int, default=3,
                        help="정상 샘플 1개당 합성 이상 샘플 수 (기본 3)")
    parser.add_argument("--strength", type=float, default=0.3,
                        help="변형 강도 (기본 0.3, 어깨너비 단위)")
    parser.add_argument("--exercises", nargs="+", default=EXERCISES)
    args = parser.parse_args()

    generate_synthetic(
        tier3_root=args.tier3_root,
        output_dir=args.output_dir,
        n_per_normal=args.n_per_normal,
        strength=args.strength,
        exercises=args.exercises,
    )


if __name__ == "__main__":
    main()

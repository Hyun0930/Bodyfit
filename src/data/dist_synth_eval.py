"""학습 데이터 분포 기반 정상/이상 레이블 생성 + 합성 이상 포즈 생성.

접근법:
  1. 학습 데이터(processed/) 전체의 관절별 통계(μ, σ) 계산
  2. Tier3 샘플의 OOD 점수 = 학습 분포와의 평균 z-score
     → OOD < threshold → 정상(label=0), 분포 안에 있는 샘플
  3. 합성 이상: 정상 샘플 keypoint를 k·σ 만큼 비정상 방향으로 이동
     (k=2.5~3.0 → 학습 분포 밖, 그러나 인간 관절 가동 범위 내)

왜 기존 GPT-4o 레이블이 부정확했나:
  - YouTube 영상은 대부분 정상 자세(선택 편향) → GPT-4o가 미세 오류를 억지로 찾음
  - Flow 학습 분포 = 전문가 경기/지도 영상 다양한 스타일
  - GPT-4o "정상" = 교과서 폼 → 학습 분포와 오히려 차이 가능
  - GPT-4o "이상" = 파워리프팅 내 미세 오류 → 여전히 학습 분포 안에 있음

Usage:
    python3 -m src.data.dist_synth_eval \
        --processed_root /home/work/body_fit/processed \
        --tier3_root /home/work/body_fit/test \
        --output_dir /home/work/body_fit/test_dist_synth \
        --n_train_sample 3000 \
        --perturb_k 2.5
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

# BlazePose 33 keypoint — 좌우 대칭 쌍
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16

# 상체 관절 (어깨 위쪽)
UPPER_BODY_JOINTS = list(range(0, 23))  # 0~22: 얼굴/상체

# 왼쪽/오른쪽 관절
LEFT_JOINTS = [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
RIGHT_JOINTS = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]


# ──────────────────────────────────────────────────
# 1. 학습 분포 통계 계산
# ──────────────────────────────────────────────────

def compute_train_stats(processed_root: Path, n_sample: int = 3000, seed: int = 42):
    """학습 데이터에서 관절별 (μ, σ) 계산.

    Returns:
        mean: (33, 3)
        std:  (33, 3)  — 0이면 1e-6으로 clamp
    """
    rng = np.random.default_rng(seed)
    all_files = []
    for ex in EXERCISES:
        ex_dir = processed_root / ex
        if ex_dir.exists():
            all_files.extend(sorted(ex_dir.glob("*.npz")))

    if not all_files:
        raise FileNotFoundError(f"processed_root에 npz 없음: {processed_root}")

    # 서브샘플
    if len(all_files) > n_sample:
        idx = rng.choice(len(all_files), n_sample, replace=False)
        all_files = [all_files[i] for i in idx]

    print(f"학습 통계 계산 중 ({len(all_files)}개 파일)...")

    frames_list = []
    for p in all_files:
        try:
            d = np.load(p, allow_pickle=True)
            body = d["body"]
            if np.any(np.abs(body) > 100):
                continue
            pose = d["pose"]  # (64, 33, 3)
            # 프레임 평균으로 압축 → (33, 3)
            frames_list.append(pose.mean(axis=0))
        except Exception:
            continue

    if not frames_list:
        raise RuntimeError("유효한 학습 데이터 없음")

    arr = np.stack(frames_list, axis=0)  # (N, 33, 3)
    mean = arr.mean(axis=0)              # (33, 3)
    std = arr.std(axis=0)                # (33, 3)
    std = np.where(std < 1e-6, 1e-6, std)
    print(f"  학습 데이터 {arr.shape[0]}개, 관절 범위 확인:")
    print(f"  전체 pose mean range: {mean.min():.3f} ~ {mean.max():.3f}")
    print(f"  전체 pose std  range: {std.min():.4f} ~ {std.max():.4f}")
    return mean, std


# ──────────────────────────────────────────────────
# 2. Tier3 OOD 점수 계산
# ──────────────────────────────────────────────────

def ood_score(pose: np.ndarray, train_mean: np.ndarray, train_std: np.ndarray) -> float:
    """프레임 평균 포즈의 관절별 z-score 평균.

    낮을수록 학습 분포에 가까움 (정상에 가까움).
    """
    p_mean = pose.mean(axis=0)           # (33, 3)
    z = np.abs((p_mean - train_mean) / train_std)  # (33, 3)
    return float(z.mean())


# ──────────────────────────────────────────────────
# 3. 변형 함수 — k·σ 기반
# ──────────────────────────────────────────────────

def perturb_knee_valgus(pose: np.ndarray, train_std: np.ndarray, k: float) -> np.ndarray:
    """무릎 내반: 좌우 무릎을 k·σ_knee_x 만큼 중앙으로."""
    out = pose.copy()
    sigma_x = train_std[LEFT_KNEE, 0]   # 무릎 x 방향 학습 분산
    shift = k * sigma_x
    out[:, LEFT_KNEE, 0] -= shift        # 왼무릎 → 오른쪽 (내반)
    out[:, RIGHT_KNEE, 0] += shift       # 오른무릎 → 왼쪽 (내반)
    return out


def perturb_forward_lean(pose: np.ndarray, train_std: np.ndarray, k: float) -> np.ndarray:
    """과도한 앞 기울기: 상체를 k·σ_z 만큼 z방향(깊이)으로."""
    out = pose.copy()
    sigma_z = train_std[LEFT_SHOULDER, 2]   # 어깨 z방향 학습 분산
    shift = k * sigma_z
    out[:, UPPER_BODY_JOINTS, 2] += shift
    return out


def perturb_asymmetry(pose: np.ndarray, train_std: np.ndarray, k: float) -> np.ndarray:
    """좌우 비대칭: 왼쪽 관절을 k·σ_x 만큼 lateral shift."""
    out = pose.copy()
    sigma_x = train_std[LEFT_SHOULDER, 0]
    shift = k * sigma_x
    out[:, LEFT_JOINTS, 0] -= shift
    return out


def perturb_hip_shift(pose: np.ndarray, train_std: np.ndarray, k: float) -> np.ndarray:
    """고관절 전방 이동: 힙을 앞으로 밀기 (버트윙크/과신전 시뮬레이션)."""
    out = pose.copy()
    sigma_z = train_std[LEFT_HIP, 2]
    shift = k * sigma_z
    out[:, [LEFT_HIP, RIGHT_HIP], 2] += shift
    out[:, [LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE], 2] -= shift * 0.5
    return out


PERTURB_FNS = [
    ("knee_valgus", perturb_knee_valgus),
    ("forward_lean", perturb_forward_lean),
    ("asymmetry", perturb_asymmetry),
    ("hip_shift", perturb_hip_shift),
]


# ──────────────────────────────────────────────────
# 4. 메인 생성 함수
# ──────────────────────────────────────────────────

def build_dist_synth_eval(
    processed_root: Path,
    tier3_root: Path,
    output_dir: Path,
    n_train_sample: int = 3000,
    ood_threshold: float = 1.5,
    perturb_k: float = 2.5,
    n_per_normal: int = 4,
    exercises: list = None,
):
    """Tier3 분포 분석 + 합성 이상 생성 → labels.json 출력.

    Args:
        ood_threshold: 이 z-score 이하면 정상으로 분류
        perturb_k: 합성 이상 생성 시 k·σ 배율 (2.5 = 학습 분포 2.5σ 밖)
        n_per_normal: 정상 샘플 1개당 합성 이상 수
    """
    exercises = exercises or EXERCISES
    output_dir.mkdir(parents=True, exist_ok=True)

    # 학습 통계
    train_mean, train_std = compute_train_stats(processed_root, n_sample=n_train_sample)
    np.save(output_dir / "train_mean.npy", train_mean)
    np.save(output_dir / "train_std.npy", train_std)

    # Tier3 OOD 점수 계산
    all_tier3 = []
    for ex in exercises:
        ex_dir = tier3_root / ex
        if not ex_dir.exists():
            continue
        for p in sorted(ex_dir.glob("*.npz")):
            try:
                d = np.load(p, allow_pickle=True)
                pose = d["pose"]
                body = d["body"]
                if np.any(np.abs(body) > 100):
                    continue
                score = ood_score(pose, train_mean, train_std)
                all_tier3.append((ex, p, pose, body, score))
            except Exception as e:
                print(f"  건너뜀: {p.name} ({e})")

    scores = np.array([s for _, _, _, _, s in all_tier3])
    print(f"\nTier3 OOD 점수 통계:")
    print(f"  N={len(scores)}, mean={scores.mean():.3f}, std={scores.std():.3f}")
    print(f"  min={scores.min():.3f}, max={scores.max():.3f}")
    print(f"  threshold={ood_threshold} → 정상 후보: {(scores < ood_threshold).sum()}개")

    # 정상 선별 (OOD 점수 < threshold)
    normal_samples = [(ex, p, pose, body) for ex, p, pose, body, s in all_tier3 if s < ood_threshold]
    print(f"  정상으로 분류: {len(normal_samples)}개")

    if len(normal_samples) == 0:
        # threshold 자동 조정: 하위 40%
        thresh_auto = float(np.percentile(scores, 40))
        print(f"  임계값 자동 조정 → {thresh_auto:.3f} (하위 40%)")
        normal_samples = [(ex, p, pose, body) for ex, p, pose, body, s in all_tier3 if s < thresh_auto]

    labels = {}

    # 정상 샘플 복사 + labels
    for ex, p, pose, body in normal_samples:
        dst_dir = output_dir / ex
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        key = f"{ex}/{p.stem}"
        labels[key] = {"label": 0, "source": "tier3_in_dist"}

    # 합성 이상 생성
    perturb_cycle = PERTURB_FNS[:n_per_normal]
    n_synth = 0
    for ex, p, pose, body in normal_samples:
        dst_dir = output_dir / ex
        for pname, pfn in perturb_cycle:
            perturbed = pfn(pose, train_std, k=perturb_k)
            stem = f"{p.stem}_synth_{pname}"
            dst = dst_dir / f"{stem}.npz"
            np.savez(dst, pose=perturbed.astype(np.float32), body=body.astype(np.float32))
            # 합성 이상의 OOD 점수 확인
            synth_score = ood_score(perturbed, train_mean, train_std)
            labels[f"{ex}/{stem}"] = {
                "label": 1,
                "source": "synthetic",
                "perturbation": pname,
                "perturb_k": perturb_k,
                "ood_score": round(synth_score, 4),
            }
            n_synth += 1

    # OOD 점수 검증
    synth_scores = [v["ood_score"] for v in labels.values() if v.get("source") == "synthetic"]
    if synth_scores:
        print(f"\n합성 이상 OOD 점수: mean={np.mean(synth_scores):.3f}, min={np.min(synth_scores):.3f}")
        print(f"  → 학습 분포 대비 평균 {np.mean(synth_scores):.1f}σ 이탈")

    out_labels = output_dir / "labels.json"
    out_labels.write_text(json.dumps(labels, ensure_ascii=False, indent=2))

    n_normal = sum(1 for v in labels.values() if v["label"] == 0)
    n_abnormal = sum(1 for v in labels.values() if v["label"] == 1)
    print(f"\n생성 완료: 정상 {n_normal}개, 합성 이상 {n_abnormal}개")
    print(f"저장 위치: {output_dir}")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True, type=Path,
                        help="학습 processed/ 루트 (processed/squat/*.npz 등)")
    parser.add_argument("--tier3_root", required=True, type=Path,
                        help="Tier3 test/ 루트")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="출력 디렉토리 (새 test_dist_synth/)")
    parser.add_argument("--n_train_sample", type=int, default=3000,
                        help="학습 통계 계산에 사용할 파일 수 (기본 3000)")
    parser.add_argument("--ood_threshold", type=float, default=1.5,
                        help="Tier3 정상 판별 OOD 임계값 (기본 1.5σ)")
    parser.add_argument("--perturb_k", type=float, default=2.5,
                        help="합성 이상 이탈 강도 (k·σ, 기본 2.5)")
    parser.add_argument("--n_per_normal", type=int, default=4,
                        help="정상 샘플당 합성 이상 수 (기본 4)")
    parser.add_argument("--exercises", nargs="+", default=EXERCISES)
    args = parser.parse_args()

    build_dist_synth_eval(
        processed_root=args.processed_root,
        tier3_root=args.tier3_root,
        output_dir=args.output_dir,
        n_train_sample=args.n_train_sample,
        ood_threshold=args.ood_threshold,
        perturb_k=args.perturb_k,
        n_per_normal=args.n_per_normal,
        exercises=args.exercises,
    )


if __name__ == "__main__":
    main()

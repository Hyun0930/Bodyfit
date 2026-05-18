"""
keypoint .npy → 정규화 + rep 분할 + 64프레임 resample → .npz

Usage:
    python src/data/preprocess.py --exercise squat
    python src/data/preprocess.py --exercise all
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from body_feature import extract_body_feature

ROOT = Path(__file__).resolve().parents[2]
KEYPOINTS_DIR = ROOT / "data" / "keypoints"
PROCESSED_DIR = ROOT / "data" / "processed"

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

# (joint_a, vertex, joint_b) — 각도 계산 기준
ANGLE_JOINTS = {
    "squat":    (23, 25, 27),   # 고관절-무릎-발목
    "bench":    (11, 13, 15),   # 어깨-팔꿈치-손목
    "deadlift": (11, 23, 25),   # 어깨-고관절-무릎
    "ohp":      (13, 11, 23),   # 팔꿈치-어깨-고관절
}

TARGET_FRAMES = 64
MIN_REP_FRAMES = 20


def compute_angle(kps: np.ndarray, a: int, v: int, b: int) -> np.ndarray:
    """관절 각도 시계열 반환 (도 단위). kps: (T, 33, 3 or 4)"""
    va = kps[:, a, :2] - kps[:, v, :2]
    vb = kps[:, b, :2] - kps[:, v, :2]
    cos = (va * vb).sum(axis=1) / (
        np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-6
    )
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def normalize(kps: np.ndarray) -> np.ndarray:
    """
    Hip-center 이동 + 어깨너비 1.0 스케일.
    kps: (T, 33, 4) → returns (T, 33, 3) (x, y, z — visibility 제외)
    """
    kps = kps.copy().astype(np.float32)

    # Hip-center 이동 (keypoint 23, 24 = 좌우 고관절)
    hip_center = (kps[:, 23, :2] + kps[:, 24, :2]) / 2  # (T, 2)
    kps[:, :, :2] -= hip_center[:, None, :]

    # 어깨너비 1.0으로 스케일 (keypoint 11, 12 = 좌우 어깨)
    shoulder_w = np.linalg.norm(kps[:, 11, :2] - kps[:, 12, :2], axis=-1)  # (T,)
    scale = 1.0 / (shoulder_w.mean() + 1e-6)
    kps[:, :, :3] *= scale

    return kps[:, :, :3]  # (T, 33, 3)


def resample(rep: np.ndarray, n: int = TARGET_FRAMES) -> np.ndarray:
    """rep: (T_rep, 33, 3) → (n, 33, 3)"""
    T = rep.shape[0]
    idx_new = np.linspace(0, T - 1, n)
    idx_old = np.arange(T)
    out = np.zeros((n, 33, 3), dtype=np.float32)
    for j in range(33):
        for c in range(3):
            out[:, j, c] = np.interp(idx_new, idx_old, rep[:, j, c])
    return out


def segment_reps(kps_raw: np.ndarray, exercise: str) -> list[np.ndarray]:
    """
    kps_raw: (T, 33, 4) 원본 (정규화 전)
    returns: list of (T_rep, 33, 3) 정규화 완료된 rep 조각들
    """
    a, v, b = ANGLE_JOINTS[exercise]
    angles = compute_angle(kps_raw, a, v, b)  # (T,)

    # 관절 굴곡 최솟값(가장 구부러진 지점)을 rep 경계로 사용
    # 스쿼트는 무릎 각도 최솟값, 벤치는 팔꿈치 최솟값 등
    peaks, _ = find_peaks(-angles, prominence=10, distance=15)

    if len(peaks) < 2:
        return []

    # rep 경계: 연속된 peak 사이를 1 rep으로 정의
    kps_norm = normalize(kps_raw)  # (T, 33, 3)
    reps = []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        if end - start < MIN_REP_FRAMES:
            continue
        reps.append(kps_norm[start:end])

    return reps


def process_file(npy_path: Path, exercise: str, out_dir: Path) -> int:
    """단일 .npy 파일 처리. 저장된 rep 수 반환."""
    kps_raw = np.load(npy_path)  # (T, 33, 4)
    if kps_raw.ndim != 3 or kps_raw.shape[1:] != (33, 4):
        return 0

    body = extract_body_feature(kps_raw)
    reps = segment_reps(kps_raw, exercise)

    video_id = npy_path.stem
    saved = 0
    for i, rep in enumerate(reps):
        pose = resample(rep)  # (64, 33, 3)
        meta = {
            "video_id": video_id,
            "rep_idx": i,
            "exercise": exercise,
            "n_original_frames": int(rep.shape[0]),
        }
        out_path = out_dir / f"{video_id}_rep{i:02d}.npz"
        np.savez_compressed(out_path, pose=pose, body=body, meta=json.dumps(meta))
        saved += 1

    return saved


def process_exercise(exercise: str) -> None:
    kp_dir = KEYPOINTS_DIR / exercise
    out_dir = PROCESSED_DIR / exercise
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(kp_dir.glob("*.npy"))
    if not npy_files:
        print(f"[{exercise}] no .npy files found in {kp_dir}")
        return

    total = 0
    for f in npy_files:
        n = process_file(f, exercise, out_dir)
        total += n
        print(f"  {f.name} → {n} reps")

    print(f"[{exercise}] done — {total} reps saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", choices=EXERCISES + ["all"], default="all")
    args = parser.parse_args()

    targets = EXERCISES if args.exercise == "all" else [args.exercise]
    for ex in targets:
        process_exercise(ex)


if __name__ == "__main__":
    main()

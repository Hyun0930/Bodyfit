"""
영상 → MediaPipe BlazePose → .npy keypoint 저장

Usage:
    python src/data/extract_keypoints.py --exercise squat
    python src/data/extract_keypoints.py --exercise all
    python src/data/extract_keypoints.py --exercise squat --video path/to/video.mp4  # 단일 테스트

MediaPipe 0.10+ Tasks API 사용 — models_mediapipe/pose_landmarker_heavy.task 필요
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
KEYPOINTS_DIR = ROOT / "data" / "keypoints"
MODEL_PATH = ROOT / "models_mediapipe" / "pose_landmarker_heavy.task"

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

# 전체 프레임 중 이 비율 이상이 저시인성이면 영상 스킵
VIS_FAIL_THRESHOLD = 0.50
VIS_MIN = 0.5


class OneEuroFilter:
    """1€ filter — 빠른 움직임은 덜 스무딩, 정지 시 더 스무딩."""

    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, beta: float = 0.0, dcutoff: float = 1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = None
        self._dx = 0.0

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x is None:
            self._x = x
            return x
        dx = (x - self._x) * self.freq
        edx = self._alpha(self.dcutoff) * dx + (1 - self._alpha(self.dcutoff)) * self._dx
        cutoff = self.mincutoff + self.beta * abs(edx)
        self._x = self._alpha(cutoff) * x + (1 - self._alpha(cutoff)) * self._x
        self._dx = edx
        return self._x


def _build_landmarker() -> mp_vision.PoseLandmarker:
    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_keypoints(video_path: Path) -> np.ndarray | None:
    """
    단일 영상에서 keypoint 추출.
    Returns:
        (T, 33, 4) — x, y, z, visibility  또는 품질 미달 시 None
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    frame_idx = 0

    with _build_landmarker() as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                lms = result.pose_landmarks[0]  # NormalizedLandmark list
                kp = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                    dtype=np.float32,
                )  # (33, 4)
            else:
                kp = np.full((33, 4), np.nan, dtype=np.float32)

            frames.append(kp)
            frame_idx += 1

    cap.release()

    if not frames:
        return None

    kps = np.stack(frames)  # (T, 33, 4)

    # 품질 필터: visibility < 0.5인 프레임 비율
    low_vis = (kps[:, :, 3] < VIS_MIN).mean()
    if low_vis > VIS_FAIL_THRESHOLD:
        return None

    # visibility < 0.5인 개별 keypoint → NaN
    mask = kps[:, :, 3] < VIS_MIN  # (T, 33)
    kps[mask] = np.nan

    # 선형 보간 (각 keypoint·채널 독립적으로)
    kps = _interpolate_nan(kps)

    # 1€ filter (x, y만 — z, visibility는 그대로)
    kps = _apply_euro_filter(kps, fps)

    return kps


def _interpolate_nan(kps: np.ndarray) -> np.ndarray:
    """NaN 프레임을 선형 보간으로 채운다. kps: (T, 33, 4)"""
    T = kps.shape[0]
    t = np.arange(T, dtype=float)
    for j in range(33):
        for c in range(4):
            col = kps[:, j, c]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                kps[:, j, c] = 0.0
            elif not valid.all():
                kps[:, j, c] = np.interp(t, t[valid], col[valid])
    return kps


def _apply_euro_filter(kps: np.ndarray, fps: float) -> np.ndarray:
    """x, y 채널에 1€ filter 적용. kps: (T, 33, 4)"""
    out = kps.copy()
    for j in range(33):
        for c in range(2):  # x, y만
            f = OneEuroFilter(freq=fps)
            for t in range(kps.shape[0]):
                out[t, j, c] = f(kps[t, j, c])
    return out


def process_exercise(exercise: str) -> None:
    raw_dir = RAW_DIR / exercise
    out_dir = KEYPOINTS_DIR / exercise
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(raw_dir.glob("*.mp4"))
    if not videos:
        print(f"[{exercise}] no .mp4 files in {raw_dir}")
        return

    failed = []
    saved = 0
    for vp in videos:
        out_path = out_dir / f"{vp.stem}.npy"
        if out_path.exists():
            continue  # 이미 처리됨

        kps = extract_keypoints(vp)
        if kps is None:
            failed.append(vp.name)
            print(f"  SKIP {vp.name} (low visibility)")
        else:
            np.save(out_path, kps)
            saved += 1
            print(f"  OK   {vp.name} → {kps.shape}")

    if failed:
        (out_dir / "failed.txt").write_text("\n".join(failed))

    print(f"[{exercise}] saved={saved}  failed={len(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", choices=EXERCISES + ["all"], default="all")
    parser.add_argument("--video", type=Path, default=None, help="단일 영상 테스트용")
    args = parser.parse_args()

    if args.video:
        kps = extract_keypoints(args.video)
        if kps is None:
            print("FAILED — low visibility or no pose detected")
        else:
            print(f"OK — shape: {kps.shape}")
            # keypoints 디렉토리에 저장 (exercise 추론: 부모 디렉토리명 사용)
            exercise_guess = args.video.parent.name
            if exercise_guess in EXERCISES:
                out_dir = KEYPOINTS_DIR / exercise_guess
            else:
                out_dir = KEYPOINTS_DIR / "test"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{args.video.stem}.npy"
            np.save(out_path, kps)
            print(f"saved → {out_path}")
        return

    targets = EXERCISES if args.exercise == "all" else [args.exercise]
    for ex in targets:
        process_exercise(ex)


if __name__ == "__main__":
    main()

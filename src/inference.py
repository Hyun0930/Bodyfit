"""단일 영상 → 전체 파이프라인 inference.

Usage:
    from src.inference import BodyFitInference
    engine = BodyFitInference(exercise="squat")
    results = engine.run("video.mp4")
"""
import os
from pathlib import Path

import numpy as np
import torch

from src.data.extract_keypoints import extract_keypoints
from src.data.preprocess import segment_reps, resample
from src.data.body_feature import extract_body_feature
from src.models.cvae import CVAE
from src.evaluation.llm_feedback import feedback_from_model_output, heatmap_to_top_joints

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

DEFAULT_CKPT = Path(os.environ.get("BODYFIT_DATA", "data")) / ".." / "checkpoints" / "cvae" / "best.pt"


class BodyFitInference:
    def __init__(
        self,
        exercise: str = "squat",
        ckpt_path: str | Path | None = None,
        device: str | None = None,
    ):
        assert exercise in EXERCISES, f"exercise must be one of {EXERCISES}"
        self.exercise = exercise

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = CVAE().to(self.device)
        state = ckpt.get("model_state", ckpt)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.body_mean = ckpt.get("body_mean")
        self.body_std = ckpt.get("body_std")
        if self.body_mean is not None:
            self.body_mean = self.body_mean.to(self.device)
            self.body_std = self.body_std.to(self.device)

        # threshold: val set 95th percentile 저장된 경우 사용, 없으면 None
        self.threshold = ckpt.get("threshold_95", None)

    def run(self, video_path: str | Path) -> list[dict]:
        """영상 → rep별 결과 리스트 반환.

        Returns:
            list of dict:
                rep_idx, anomaly_score, threshold, is_anomaly,
                top_joints, feedback, heatmap (numpy)
        """
        video_path = Path(video_path)

        # 1. MediaPipe 포즈 추출
        kps = extract_keypoints(video_path)  # (T, 33, 4)
        if kps is None or len(kps) < 20:
            raise ValueError("포즈 추출 실패 또는 영상이 너무 짧습니다.")

        # 2. body feature (전체 영상에서 한 번 추출)
        body = extract_body_feature(kps)  # (7,)
        if np.any(np.abs(body) > 100):
            body = np.clip(body, -10, 10)

        # 3. rep 분할 (segment_reps 내부에서 normalize 처리)
        reps = segment_reps(kps, self.exercise)  # list of (frames, 33, 3)
        if not reps:
            raise ValueError("rep을 감지하지 못했습니다. 영상에 동작이 충분한지 확인하세요.")

        # 5. 각 rep 추론
        body_t = torch.tensor(body, dtype=torch.float32).unsqueeze(0).to(self.device)
        if self.body_mean is not None:
            body_t = (body_t - self.body_mean) / self.body_std

        results = []
        for i, (rep, _, _) in enumerate(reps):
            pose_64 = resample(rep)  # (64, 33, 3)
            pose_t = torch.tensor(pose_64, dtype=torch.float32).unsqueeze(0).to(self.device)

            score = self.model.anomaly_score(pose_t, body_t).item()
            heatmap = self.model.joint_attribution(pose_t, body_t).squeeze(0).cpu().numpy()

            thr = self.threshold if self.threshold is not None else score * 1.2
            top_joints = heatmap_to_top_joints(heatmap, top_k=3)

            fb = feedback_from_model_output(
                exercise=self.exercise,
                anomaly_score=score,
                threshold=thr,
                heatmap=heatmap,
                top_k=3,
            )

            results.append({
                "rep_idx": i + 1,
                "anomaly_score": round(score, 4),
                "threshold": round(thr, 4),
                "is_anomaly": fb["is_anomaly"],
                "top_joints": top_joints,
                "feedback": fb["feedback"],
                "heatmap": heatmap,
            })

        return results

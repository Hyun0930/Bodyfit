"""Claude API 연동 자세 피드백 생성."""

from __future__ import annotations

import os

import anthropic
import numpy as np

# MediaPipe BlazePose 33 keypoint 한국어 이름
JOINT_NAMES: dict[int, str] = {
    0: "코",
    1: "왼눈 안쪽",
    2: "왼눈",
    3: "왼눈 바깥쪽",
    4: "오른눈 안쪽",
    5: "오른눈",
    6: "오른눈 바깥쪽",
    7: "왼귀",
    8: "오른귀",
    9: "입 왼쪽",
    10: "입 오른쪽",
    11: "왼어깨",
    12: "오른어깨",
    13: "왼팔꿈치",
    14: "오른팔꿈치",
    15: "왼손목",
    16: "오른손목",
    17: "왼손 엄지",
    18: "오른손 엄지",
    19: "왼손 검지",
    20: "오른손 검지",
    21: "왼손 새끼",
    22: "오른손 새끼",
    23: "왼엉덩이",
    24: "오른엉덩이",
    25: "왼무릎",
    26: "오른무릎",
    27: "왼발목",
    28: "오른발목",
    29: "왼발 뒤꿈치",
    30: "오른발 뒤꿈치",
    31: "왼발 앞",
    32: "오른발 앞",
}

EXERCISE_KO: dict[str, str] = {
    "squat": "스쿼트",
    "bench": "벤치프레스",
    "deadlift": "데드리프트",
    "ohp": "오버헤드프레스",
}

# 얼굴/손 관절 — 자세 피드백과 무관하므로 제외
_FACE_HAND_JOINTS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20, 21, 22}


def heatmap_to_top_joints(
    heatmap: np.ndarray,
    top_k: int = 3,
    exclude_face: bool = True,
) -> list[str]:
    """관절 attribution heatmap에서 상위 k개 관절 이름 반환.

    Args:
        heatmap: (T, 33) 또는 (33,) — attribution 점수
        top_k:   반환할 관절 수
        exclude_face: 얼굴/손 관절 제외 여부

    Returns:
        상위 관절 한국어 이름 리스트
    """
    heatmap = np.asarray(heatmap, dtype=float)
    if heatmap.ndim == 2:
        heatmap = heatmap.mean(axis=0)  # 시간 평균 → (33,)

    if exclude_face:
        mask = np.ones(33, dtype=bool)
        for idx in _FACE_HAND_JOINTS:
            mask[idx] = False
        heatmap = heatmap * mask

    top_indices = np.argsort(heatmap)[::-1][:top_k]
    return [JOINT_NAMES[int(i)] for i in top_indices]


def generate_feedback(
    exercise: str,
    anomaly_score: float,
    threshold: float,
    top_joints: list[str],
    model: str = "claude-sonnet-4-6",
    api_key: str | None = None,
) -> str:
    """Claude API로 자세 교정 피드백 생성.

    Args:
        exercise:      운동 종류 ("squat" | "bench" | "deadlift" | "ohp")
        anomaly_score: -log P(pose|body) 값
        threshold:     val set 5th percentile 기준값
        top_joints:    문제 관절 이름 리스트
        model:         Claude 모델 ID
        api_key:       None이면 ANTHROPIC_API_KEY 환경변수 사용

    Returns:
        한국어 피드백 문자열 (2~3문장)
    """
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    exercise_ko = EXERCISE_KO.get(exercise, exercise)
    ratio = anomaly_score / threshold if threshold > 0 else 1.0
    severity = "경미한" if ratio < 1.2 else "중간 수준의" if ratio < 1.5 else "심각한"

    prompt = (
        f"운동 종류: {exercise_ko}\n"
        f"자세 이상 점수: {anomaly_score:.1f} (기준값 {threshold:.1f}, {ratio:.0%} 수준 — {severity} 이상)\n"
        f"주요 문제 부위: {', '.join(top_joints)}\n\n"
        f"위 정보를 바탕으로 운동 자세 교정 피드백을 한국어 2~3문장으로 작성하세요. "
        f"구체적인 교정 방법을 포함하고, 전문 용어는 쉽게 풀어서 설명하세요."
    )

    message = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def feedback_from_model_output(
    exercise: str,
    anomaly_score: float,
    threshold: float,
    heatmap: np.ndarray,
    top_k: int = 3,
    **kwargs,
) -> dict:
    """모델 출력 → 피드백 생성 one-shot 편의 함수.

    Args:
        exercise:      운동 종류
        anomaly_score: 이상 점수 (scalar)
        threshold:     기준값
        heatmap:       (T, 33) joint attribution
        top_k:         상위 관절 수

    Returns:
        dict with keys: feedback, top_joints, anomaly_score, threshold, is_anomaly
    """
    top_joints = heatmap_to_top_joints(heatmap, top_k=top_k)
    is_anomaly = anomaly_score > threshold
    feedback = generate_feedback(exercise, anomaly_score, threshold, top_joints, **kwargs)
    return {
        "feedback": feedback,
        "top_joints": top_joints,
        "anomaly_score": round(float(anomaly_score), 4),
        "threshold": round(float(threshold), 4),
        "is_anomaly": bool(is_anomaly),
    }

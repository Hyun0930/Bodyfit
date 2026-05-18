import numpy as np


# MediaPipe BlazePose keypoint indices
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_HIP, _R_HIP = 23, 24
_L_KNEE, _R_KNEE = 25, 26
_L_ANKLE, _R_ANKLE = 27, 28
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_NOSE = 0


def _dist(kp: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(kp[a, :2] - kp[b, :2]))


def _dist_mid(kp: np.ndarray, a1: int, a2: int, b1: int, b2: int) -> float:
    mid_a = (kp[a1, :2] + kp[a2, :2]) / 2
    mid_b = (kp[b1, :2] + kp[b2, :2]) / 2
    return float(np.linalg.norm(mid_a - mid_b))


def _select_static_frame(kps: np.ndarray) -> np.ndarray:
    """영상 중간 10~30% 구간에서 visibility 합이 가장 높은 프레임 반환."""
    T = kps.shape[0]
    lo, hi = int(T * 0.10), int(T * 0.30)
    hi = max(hi, lo + 1)
    segment = kps[lo:hi]  # (S, 33, 4)
    vis_sum = segment[:, :, 3].sum(axis=1)  # (S,)
    best = int(np.argmax(vis_sum))
    return segment[best]  # (33, 4)


def extract_body_feature(kps: np.ndarray) -> np.ndarray:
    """
    Args:
        kps: (T, 33, 4) — x, y, z, visibility (MediaPipe 정규화 좌표)
    Returns:
        body_vec: (7,) — scale-invariant 체형 비율 벡터
    """
    kp = _select_static_frame(kps)  # (33, 4)
    eps = 1e-6

    b = np.zeros(7, dtype=np.float32)

    # b[0] thigh/shin — 대퇴/경골 비율 (스쿼트·데드 자세 기준 핵심)
    thigh = (_dist(kp, _L_HIP, _L_KNEE) + _dist(kp, _R_HIP, _R_KNEE)) / 2
    shin  = (_dist(kp, _L_KNEE, _L_ANKLE) + _dist(kp, _R_KNEE, _R_ANKLE)) / 2
    b[0] = thigh / (shin + eps)

    # b[1] torso/leg — 상체/하체 비율
    torso = _dist_mid(kp, _L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP)
    leg   = _dist_mid(kp, _L_HIP, _R_HIP, _L_ANKLE, _R_ANKLE)
    b[1] = torso / (leg + eps)

    # b[2] arm/torso — 팔 길이/몸통 비율 (벤치·데드·OHP 그립 위치)
    arm = (_dist(kp, _L_SHOULDER, _L_WRIST) + _dist(kp, _R_SHOULDER, _R_WRIST)) / 2
    b[2] = arm / (torso + eps)

    # b[3] shoulder/hip width — V-taper 반영
    b[3] = _dist(kp, _L_SHOULDER, _R_SHOULDER) / (_dist(kp, _L_HIP, _R_HIP) + eps)

    # b[4] armspan/height — 윙스팬/키 (벤치프레스 ROM에 직접 영향)
    armspan = _dist(kp, _L_WRIST, _R_WRIST)
    height  = (_dist(kp, _NOSE, _L_ANKLE) + _dist(kp, _NOSE, _R_ANKLE)) / 2
    b[4] = armspan / (height + eps)

    # b[5] pelvic_tilt — 골반 좌우 기울기 (자연스러운 중립 자세 기준)
    # hip-center 기준으로 정규화
    hip_width = _dist(kp, _L_HIP, _R_HIP)
    b[5] = (kp[_L_HIP, 1] - kp[_R_HIP, 1]) / (hip_width + eps)

    # b[6] L/R symmetry — 비대칭 체형(측만증 등) 보정용
    # hip-center 기준 좌우 무릎 x좌표 대칭도
    hip_cx = (kp[_L_HIP, 0] + kp[_R_HIP, 0]) / 2
    knee_width = _dist(kp, _L_KNEE, _R_KNEE)
    b[6] = abs((kp[_L_KNEE, 0] - hip_cx) + (kp[_R_KNEE, 0] - hip_cx)) / (knee_width + eps)

    return b

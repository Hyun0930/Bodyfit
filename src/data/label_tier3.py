"""
Tier 3 평가셋 Vision API 라벨링 스크립트.

각 rep의 원본 mp4에서 8장 프레임을 추출하여 Claude Vision API로 분석,
labels_draft.json 초안을 생성한다. confidence=low 항목은 수동 검수 필요.

Usage:
    python -m src.data.label_tier3 \
        --data_root data/test \
        --video_root data/raw/tier3
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import anthropic
import cv2
import numpy as np

EXERCISES = ["squat", "bench", "deadlift", "ohp"]
N_FRAMES = 8  # rep당 추출할 프레임 수
MODEL = "claude-sonnet-4-6"

PROMPT_TEMPLATE = """당신은 파워리프팅 전문 코치입니다.
아래 이미지들은 {exercise} 동작 1회(rep)를 시간 순서대로 캡처한 {n}장의 프레임입니다.

다음 기준으로 분석하세요:
- 정상(label=0): 올바른 자세, 큰 문제 없음
- 이상(label=1): 자세 오류 존재 (예: 무릎 내반, 허리 라운딩, 깊이 부족, 좌우 비대칭 등)

반드시 아래 JSON 형식으로만 응답하세요:
{{
  "label": 0 또는 1,
  "joints": ["문제 관절 목록, 정상이면 빈 배열"],
  "reason": "판단 이유 1~2문장",
  "confidence": "high 또는 low"
}}

confidence=low는 판단이 애매하거나 카메라 각도·화질 문제로 확신하기 어려운 경우."""

EXERCISE_KO = {
    "squat": "스쿼트",
    "bench": "벤치프레스",
    "deadlift": "데드리프트",
    "ohp": "오버헤드프레스",
}


def extract_frames(video_path: Path, start: int, end: int, n: int = N_FRAMES) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frame_indices = np.linspace(start, end, n, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def frame_to_b64(frame: np.ndarray, max_size: int = 512) -> str:
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.standard_b64encode(buf).decode()


def label_rep(
    client: anthropic.Anthropic,
    frames: list[np.ndarray],
    exercise: str,
) -> dict:
    content = []
    for frame in frames:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": frame_to_b64(frame),
            },
        })
    content.append({
        "type": "text",
        "text": PROMPT_TEMPLATE.format(
            exercise=EXERCISE_KO.get(exercise, exercise),
            n=len(frames),
        ),
    })

    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": content}],
    )
    text = resp.content[0].text.strip()

    # JSON 파싱 (마크다운 코드블록 제거)
    if "```" in text:
        text = text.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"label": -1, "joints": [], "reason": text, "confidence": "low"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/test", help="Tier 3 processed npz 루트")
    parser.add_argument("--video_root", default="data/raw/tier3", help="Tier 3 mp4 루트")
    parser.add_argument("--exercise", choices=EXERCISES + ["all"], default="all")
    parser.add_argument("--out", default=None, help="출력 JSON 경로 (기본: data_root/labels_draft.json)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY 환경변수를 설정하세요.")

    client = anthropic.Anthropic(api_key=api_key)
    data_root = Path(args.data_root)
    video_root = Path(args.video_root)
    out_path = Path(args.out) if args.out else data_root / "labels_draft.json"

    # 기존 초안 로드 (재실행 시 이어서 진행)
    draft: dict = {}
    if out_path.exists():
        draft = json.loads(out_path.read_text())

    targets = EXERCISES if args.exercise == "all" else [args.exercise]

    for exercise in targets:
        npz_dir = data_root / exercise
        if not npz_dir.exists():
            print(f"[{exercise}] {npz_dir} 없음, 스킵")
            continue

        npz_files = sorted(npz_dir.glob("*.npz"))
        print(f"[{exercise}] {len(npz_files)}개 rep 처리 시작")

        for npz_path in npz_files:
            key = f"{exercise}/{npz_path.stem}"
            if key in draft:
                continue  # 이미 처리됨

            data = np.load(npz_path, allow_pickle=True)
            meta = json.loads(str(data["meta"]))

            video_id = meta["video_id"]
            start_frame = meta.get("rep_start_frame")
            end_frame = meta.get("rep_end_frame")

            if start_frame is None or end_frame is None:
                draft[key] = {"label": -1, "joints": [], "reason": "rep_start/end_frame 없음", "confidence": "low", "manual_check": True}
                continue

            # mp4 탐색 (확장자 mp4/webm 모두 시도)
            video_path = None
            for ext in ["mp4", "webm", "mkv"]:
                p = video_root / exercise / f"{video_id}.{ext}"
                if p.exists():
                    video_path = p
                    break

            if video_path is None:
                print(f"  SKIP {key}: mp4 없음")
                draft[key] = {"label": -1, "joints": [], "reason": "mp4 파일 없음", "confidence": "low", "manual_check": True}
                continue

            frames = extract_frames(video_path, start_frame, end_frame)
            if not frames:
                print(f"  SKIP {key}: 프레임 추출 실패")
                draft[key] = {"label": -1, "joints": [], "reason": "프레임 추출 실패", "confidence": "low", "manual_check": True}
                continue

            try:
                result = label_rep(client, frames, exercise)
            except Exception as e:
                print(f"  ERROR {key}: {e}")
                draft[key] = {"label": -1, "joints": [], "reason": str(e), "confidence": "low", "manual_check": True}
                time.sleep(2)
                continue

            result["manual_check"] = result.get("confidence") == "low" or result.get("label") == -1
            draft[key] = result
            print(f"  {key}: label={result['label']} confidence={result['confidence']} — {result['reason'][:60]}")

            # 중간 저장 (중단 시 손실 방지)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2))
            time.sleep(0.5)  # API rate limit 방지

    # 최종 저장 및 요약
    out_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2))

    total = len(draft)
    normal = sum(1 for v in draft.values() if v.get("label") == 0)
    abnormal = sum(1 for v in draft.values() if v.get("label") == 1)
    manual = sum(1 for v in draft.values() if v.get("manual_check"))
    print(f"\n완료: 총 {total}개 | 정상 {normal} | 이상 {abnormal} | 수동검수 필요 {manual}")
    print(f"저장: {out_path}")
    print("수동 검수 후 labels_draft.json → labels.json 으로 복사하세요.")


if __name__ == "__main__":
    main()

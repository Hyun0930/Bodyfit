"""
yt-dlp로 종목별 YouTube 영상 다운로드 + metadata.csv 저장

Usage:
    python src/data/crawl.py --exercise squat --max 300
    python src/data/crawl.py --exercise all --max 300
"""
import argparse
import csv
import os
import subprocess
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ["BODYFIT_DATA"]) if "BODYFIT_DATA" in os.environ else ROOT / "data"
RAW_DIR = _DATA_ROOT / "raw"

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

QUERIES: dict[str, list[str]] = {
    "squat": [
        "IPF squat competition",
        "Squat University tutorial",
        "Jeff Nippard squat form",
        "Starting Strength squat",
        "Juggernaut squat",
    ],
    "bench": [
        "IPF bench press competition",
        "Jeff Nippard bench press form",
        "Starting Strength bench press",
        "Juggernaut bench press",
        "Alan Thrall bench press",
    ],
    "deadlift": [
        "IPF deadlift competition",
        "Juggernaut deadlift form",
        "Starting Strength deadlift",
        "Jeff Nippard deadlift",
        "Alan Thrall deadlift",
    ],
    "ohp": [
        "IPF overhead press competition",
        "Jeff Nippard overhead press form",
        "Starting Strength press",
        "Alan Thrall overhead press",
        "Juggernaut overhead press",
    ],
}

TIER3_QUERIES: dict[str, list[str]] = {
    "squat": [
        "squat form check beginner",
        "squat bad form correction",
        "squat knee cave fix",
        "squat depth problem fix",
    ],
    "bench": [
        "bench press form check beginner",
        "bench press bad form correction",
        "bench press mistake fix",
    ],
    "deadlift": [
        "deadlift form check beginner",
        "deadlift back rounding fix",
        "deadlift bad form correction",
    ],
    "ohp": [
        "overhead press form check beginner",
        "ohp bad form correction",
        "shoulder press mistake fix",
    ],
}

# yt-dlp 포맷: 720p 이상 mp4 우선, 없으면 최선
FORMAT = "bestvideo[height>=720]+bestaudio/best[height>=720]/bestvideo+bestaudio/best"


def _load_existing(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open() as f:
        return {row["video_id"] for row in csv.DictReader(f)}


def _append_meta(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "url", "title", "duration"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def _search_video_ids(query: str, max_results: int, cookies: str | None = None) -> list[dict]:
    """yt-dlp로 검색 결과 메타데이터 수집 (다운로드 없이).

    --flat-playlist: 포맷 체크 없이 메타데이터만 추출 (duration 포함)
    duration 필터는 crawl_exercise에서 적용.
    """
    cmd = [
        "yt-dlp",
        f"ytsearch{max_results}:{query}",
        "--flat-playlist",
        "--no-warnings", "--quiet",
        "--print", '{"id":"%(id)s","title":"%(title)s","duration":%(duration)s}',
    ]
    if cookies:
        cmd += ["--cookies", cookies]
    result = subprocess.run(cmd, capture_output=True, text=True)
    items = []
    for line in result.stdout.strip().splitlines():
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return items


def download_video(video_id: str, out_dir: Path, cookies: str | None = None) -> bool:
    """영상 1개 다운로드. 성공 시 True.

    android client는 cookies와 함께 사용 불가 — 다운로드 시 cookies 미사용.
    검색(search)에서만 cookies 사용.
    """
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", FORMAT,
        "--extractor-args", "youtube:player_client=android",
        "--merge-output-format", "mp4",
        "-o", out_tmpl,
        "--no-warnings", "--quiet",
        "--no-playlist",
    ]
    # android client는 cookies 미지원 → cookies 인자 무시
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def crawl_exercise(
    exercise: str,
    max_videos: int,
    cookies: str | None = None,
    tier3: bool = False,
    out_dir: Path | None = None,
) -> None:
    if out_dir is None:
        out_dir = RAW_DIR / exercise
    else:
        out_dir = out_dir / exercise
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"

    queries = TIER3_QUERIES[exercise] if tier3 else QUERIES[exercise]
    existing = _load_existing(csv_path)
    downloaded = len(existing)

    for query in queries:
        if downloaded >= max_videos:
            break

        per_query = max(10, (max_videos - downloaded) // max(1, len(queries)))
        items = _search_video_ids(query, per_query * 2, cookies=cookies)

        for item in items:
            if downloaded >= max_videos:
                break
            vid = item["id"]
            if vid in existing:
                continue
            if float(item.get("duration") or 0) > 1800:  # 30분 이상 제외
                continue

            print(f"  [{exercise}] {downloaded+1}/{max_videos} downloading {vid} — {item['title'][:60]}")
            ok = download_video(vid, out_dir, cookies=cookies)
            if ok:
                _append_meta(csv_path, {
                    "video_id": vid,
                    "url": f"https://www.youtube.com/watch?v={vid}",
                    "title": item["title"],
                    "duration": item.get("duration", ""),
                })
                existing.add(vid)
                downloaded += 1
            else:
                print(f"  SKIP {vid} (download failed)")

    print(f"[{exercise}] done — {downloaded} videos saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", choices=EXERCISES + ["all"], default="all")
    parser.add_argument("--max", type=int, default=300, help="종목당 최대 영상 수")
    parser.add_argument("--cookies", type=str, default=None, help="쿠키 파일 경로 (Netscape 형식)")
    parser.add_argument("--tier3", action="store_true", help="Tier 3 bad-form 쿼리 사용 + mp4 보존")
    parser.add_argument("--out_dir", type=str, default=None, help="출력 디렉토리 override")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    targets = EXERCISES if args.exercise == "all" else [args.exercise]
    for ex in targets:
        crawl_exercise(ex, args.max, cookies=args.cookies, tier3=args.tier3, out_dir=out_dir)


if __name__ == "__main__":
    main()

#!/bin/bash
# 크롤링 → keypoint 추출 → 전처리 전체 파이프라인 실행
# 사용법: bash run_pipeline.sh [workers]
# 예시:   bash run_pipeline.sh 16

set -e

WORKERS=${1:-$(nproc)}   # 인자 없으면 전체 코어 사용
MAX_VIDEOS=300            # 종목당 최대 영상 수

# 쿠키 파일 경로 (YouTube 봇 감지 우회)
# vfolder에 업로드 후 자동으로 사용됨
COOKIES_PATH="${BODYFIT_DATA:-./data}/../cookies.txt"
if [ -f "$COOKIES_PATH" ]; then
    COOKIES_OPT="--cookies $COOKIES_PATH"
    echo "Cookies: $COOKIES_PATH (found)"
elif [ -f "/home/work/body_fit/cookies.txt" ]; then
    COOKIES_OPT="--cookies /home/work/body_fit/cookies.txt"
    echo "Cookies: /home/work/body_fit/cookies.txt (found)"
else
    COOKIES_OPT=""
    echo "Cookies: not found (bot detection may block downloads)"
fi

# Backend.AI: vfolder 경로를 BODYFIT_DATA로 지정
# 예) export BODYFIT_DATA=/home/user/vfolder/bodyfit_data
# 미설정 시 프로젝트 루트의 data/ 사용
if [ -n "$BODYFIT_DATA" ]; then
    echo "Data root: $BODYFIT_DATA (vfolder)"
    mkdir -p "$BODYFIT_DATA"/{raw,keypoints,processed,test}/{squat,bench,deadlift,ohp}
else
    echo "Data root: ./data (local)"
fi

echo "=== BodyFit Pipeline ==="
echo "Workers: $WORKERS"
echo "Max videos per exercise: $MAX_VIDEOS"
echo "Started: $(date)"
echo ""

# nohup으로 실행 중인지 확인 (터미널 끊겨도 계속 실행)
if [ -z "$NOHUP_ACTIVE" ]; then
    echo "[TIP] 세션이 끊겨도 계속 실행하려면:"
    echo "      NOHUP_ACTIVE=1 nohup bash run_pipeline.sh $WORKERS > pipeline.log 2>&1 &"
    echo ""
fi

# Step 1: 크롤링
echo "=== [1/3] Crawling ==="
python3 src/data/crawl.py --exercise all --max $MAX_VIDEOS $COOKIES_OPT
echo ""

# Step 2: Keypoint 추출 (멀티프로세싱)
echo "=== [2/3] Keypoint Extraction (workers=$WORKERS) ==="
python3 src/data/extract_keypoints.py --exercise all --workers $WORKERS
echo ""

# Step 3: 전처리
echo "=== [3/3] Preprocessing ==="
python3 src/data/preprocess.py --exercise all
echo ""

# 결과 요약
echo "=== Pipeline Complete: $(date) ==="
echo ""
echo "--- processed rep counts ---"
PROC_DIR="${BODYFIT_DATA:-./data}/processed"
for ex in squat bench deadlift ohp; do
    count=$(ls "$PROC_DIR/$ex/"*.npz 2>/dev/null | wc -l)
    echo "  $ex: $count reps"
done
echo ""
echo "--- disk usage ---"
du -sh "$PROC_DIR"
echo ""
echo "Download command (run on MacBook):"
echo "  scp -r <server>:<path>/data/processed ./data/"

#!/bin/bash
# Backend.AI 서버 초기 환경 세팅 스크립트
# 사용법: bash setup_server.sh

set -e
echo "=== BodyFit Server Setup ==="

# Python 버전 확인 (3.11 권장, 3.10+ 허용)
PYTHON=$(which python3)
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python: $PY_VER at $PYTHON"

# pip 업그레이드
$PYTHON -m pip install --upgrade pip -q

# PyTorch (CUDA 12.1 기준 — 서버 CUDA 버전에 맞게 수정)
echo ">>> Installing PyTorch (CUDA 12.1)..."
$PYTHON -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 -q

# 공통 패키지
echo ">>> Installing requirements..."
$PYTHON -m pip install -r requirements.txt -q

# MediaPipe pose landmarker 모델 다운로드
echo ">>> Downloading MediaPipe model..."
mkdir -p models_mediapipe
if [ ! -f models_mediapipe/pose_landmarker_heavy.task ]; then
    curl -L -o models_mediapipe/pose_landmarker_heavy.task \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" \
        --progress-bar
    echo "Model downloaded."
else
    echo "Model already exists, skipping."
fi

# 데이터 디렉토리 생성
mkdir -p data/raw/squat data/raw/bench data/raw/deadlift data/raw/ohp
mkdir -p data/keypoints/squat data/keypoints/bench data/keypoints/deadlift data/keypoints/ohp
mkdir -p data/processed/squat data/processed/bench data/processed/deadlift data/processed/ohp
mkdir -p data/test

echo ""
echo "=== Setup complete ==="
echo "CPU cores available: $(nproc)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not found)"
echo ""
echo "Next: bash run_pipeline.sh"

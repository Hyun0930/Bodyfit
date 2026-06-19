FROM python:3.11-slim

# 시스템 의존성 (MediaPipe, OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (레이어 캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] python-multipart python-dotenv

# 소스 코드 복사
COPY src/ ./src/
COPY app.py ./
COPY static/ ./static/

# MediaPipe 모델 복사 (있는 경우)
COPY models_mediapipe/ ./models_mediapipe/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

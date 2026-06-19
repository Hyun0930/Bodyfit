# BodyFit

**Body-Aware Form Assessment for Big4 Powerlifts**

체형 조건부 딥러닝 모델로 개인 맞춤형 자세 이상을 탐지하는 시스템.  
Squat · Bench Press · Deadlift · Overhead Press 4종목, rep 단위로 이상 점수 계산.

> 세종대학교 딥러닝 실습 · 2026 Spring · 24013840 이동현

---

## 핵심 아이디어

기존 자세 평가 AI는 모든 사용자에게 동일한 기준을 적용하지만, 파워리프팅은 체형(대퇴/경골 비율, 팔 길이 등)에 따라 올바른 자세 기준이 다르다. BodyFit은 체형 7차원 벡터를 조건 입력으로 받아 **P(pose | body)** 분포를 학습하고, 이상 점수 **A(x) = −log P(pose | body)** 를 계산한다.

```
이상 탐지 = Positive-only 학습 (선수·가이드 영상만 사용, 오류 라벨 불필요)
```

---

## 모델 스택

| 컴포넌트 | 역할 |
|----------|------|
| **MediaPipe BlazePose** | 관절 keypoint 추출 (33점, 학습 없음) |
| **CVAE** | Phase 1 baseline — 재구성 오류 기반 이상 점수 |
| **BC-STNF** | 핵심 모델 — ST-GCN + FiLM + Conditional RealNVP |
| **GPT-4o** | 자연어 피드백 생성 (OpenAI API) |

### BC-STNF 구조

```
pose (64,33,3) + body (7,)
        ↓
Body Encoder  7→32→16  →  c (16차원)
        ↓
ST-GCN × 2 + FiLM   feat' = feat · γ(c) + β(c)
        ↓
Temporal Mean Pooling  →  D=132
        ↓
Conditional RealNVP × 6  →  z ~ N(0,I)
        ↓
A(x) = -log P(pose|body)  +  joint attribution heatmap
```

---

## 환경 설정

```bash
# Python 3.11 필수 (MediaPipe 호환)
conda create -n bodyfit python=3.11 -y
conda activate bodyfit

# PyTorch (Mac M3 MPS / Linux CUDA 자동 선택)
pip install torch torchvision torchaudio

# 나머지 패키지
pip install -r requirements.txt

# FastAPI 서버용 (추론 데모)
pip install fastapi "uvicorn[standard]" python-multipart python-dotenv
```

### MediaPipe 모델 다운로드

```bash
mkdir -p models_mediapipe
curl -L -o models_mediapipe/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

### OpenAI API 키 설정

```bash
export OPENAI_API_KEY="sk-..."
# 또는 .env 파일에 저장
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## 디렉토리 구조

```
bodyfit/
├── src/
│   ├── data/
│   │   ├── crawl.py              # yt-dlp 크롤링
│   │   ├── extract_keypoints.py  # MediaPipe 배치 처리
│   │   ├── preprocess.py         # 정규화·rep 분할·resample
│   │   ├── body_feature.py       # 체형 7차원 벡터 추출
│   │   ├── __init__.py           # BodyFitDataset
│   │   ├── tier3_dataset.py      # Tier3 평가셋 Dataset
│   │   ├── label_tier3.py        # GPT-4o 라벨링
│   │   └── dist_synth_eval.py    # Synthetic OOD 생성·평가
│   ├── models/
│   │   ├── cvae.py               # CVAE baseline
│   │   ├── body_encoder.py       # MLP 7→32→16
│   │   ├── st_gcn.py             # ST-GCN + FiLM
│   │   ├── realnvp.py            # Conditional RealNVP
│   │   └── bc_stnf.py            # BC-STNF 통합
│   ├── training/
│   │   ├── train_cvae.py
│   │   ├── train_bc_stnf.py
│   │   └── train_ablation.py     # Ablation 변형 모델
│   ├── evaluation/
│   │   ├── metrics.py            # AUROC, PR-AUC, EER, Per-joint IoU
│   │   ├── ablation.py           # Ablation 5종
│   │   ├── llm_feedback.py       # GPT-4o 피드백
│   │   └── score_dist.py         # 점수 분포 분석
│   └── inference.py              # 단일 영상 end-to-end 추론
├── app.py                        # FastAPI 서버
├── static/index.html             # 웹 프론트엔드
├── Dockerfile
├── docker-compose.yml
├── data/                         # (gitignore — 로컬 전용)
│   ├── raw/                      # yt-dlp 원본 영상
│   ├── keypoints/                # MediaPipe 추출 결과 (T×33×4)
│   ├── processed/                # 전처리 완료 쌍 (64×33×3 + 7)
│   └── test/                     # Tier3 평가셋
├── checkpoints/                  # (gitignore — 로컬 전용)
└── results/                      # (gitignore — 로컬 전용)
```

---

## 재현 순서

### Step 1 — 데이터 수집

```bash
# 종목별 YouTube 크롤링 (yt-dlp 필요)
python -m src.data.crawl --exercise squat    --max_videos 300 --out data/raw/squat
python -m src.data.crawl --exercise bench    --max_videos 300 --out data/raw/bench
python -m src.data.crawl --exercise deadlift --max_videos 250 --out data/raw/deadlift
python -m src.data.crawl --exercise ohp      --max_videos 150 --out data/raw/ohp
```

### Step 2 — Keypoint 추출

```bash
# MediaPipe BlazePose 배치 처리 (CPU 멀티프로세싱)
python -m src.data.extract_keypoints \
    --input  data/raw/squat \
    --output data/keypoints/squat \
    --workers 8
# bench / deadlift / ohp도 동일하게 실행
```

### Step 3 — 전처리

```bash
# hip-center 정규화 + rep 분할 + 64프레임 resample + 체형 벡터 추출
python -m src.data.preprocess \
    --keypoints data/keypoints/squat \
    --output    data/processed/squat \
    --exercise  squat
# bench / deadlift / ohp도 동일하게 실행
```

### Step 4 — CVAE 학습

```bash
python -m src.training.train_cvae \
    --data_root data/processed \
    --exercise  squat \
    --epochs    50 \
    --ckpt_dir  checkpoints/cvae_squat
# bench / deadlift / ohp도 동일하게 실행
```

### Step 5 — BC-STNF 학습

```bash
python -m src.training.train_bc_stnf \
    --data_root data/processed \
    --exercise  squat \
    --epochs    30 \
    --ckpt_dir  checkpoints/bc_stnf_squat
```

### Step 6 — Ablation 변형 모델 학습

```bash
# no_cond / mlp_feat / cluster_cond / raw_flow / no_body_cvae
for VARIANT in no_cond mlp_feat cluster_cond raw_flow no_body_cvae; do
    python -m src.training.train_ablation \
        --data_root data/processed \
        --variant   $VARIANT \
        --ckpt_dir  checkpoints/$VARIANT
done
```

### Step 7 — Tier3 라벨링 (선택)

```bash
# GPT-4o로 평가셋 라벨 생성 (OPENAI_API_KEY 필요)
python -m src.data.label_tier3 \
    --tier3_dir data/test \
    --out       data/test/labels.json
```

### Step 8 — Ablation 평가

```bash
# Synthetic OOD 기반 평가
python -m src.evaluation.ablation \
    --processed_dir data/processed \
    --ckpt_dir      checkpoints \
    --out           results/ablation.json

# 점수 분포 분석
python -m src.evaluation.score_dist \
    --processed_dir data/processed \
    --ckpt_dir      checkpoints/bc_stnf_squat
```

---

## 추론 데모 (Docker)

체크포인트가 준비된 상태에서 웹 인터페이스로 영상을 업로드해 실시간 분석.

```bash
# 빌드 및 실행
OPENAI_API_KEY=sk-... docker-compose up --build

# 브라우저에서 접속
open http://localhost:8000
```

영상(mp4, 5~30초, 720p+)을 업로드하면 rep별 이상 점수 + 문제 관절 + GPT-4o 한국어 피드백이 표시된다.

### 데모 영상

`demo/` 폴더에 종목별 정상/이상 샘플 영상이 포함되어 있다. 프론트엔드 동작 확인에 바로 사용할 수 있다.

```
demo/
├── squat/normal_squat.mp4
├── bench/normal_bench.mp4
├── deadlift/abnormal_deadlift.mp4
├── ohp/normal_ohp.mp4
└── ohp/abnormal_ohp.mp4
```

### Docker 없이 직접 실행

```bash
# 체크포인트 경로 지정 후 서버 실행
OPENAI_API_KEY=sk-... uvicorn app:app --host 0.0.0.0 --port 8000
```

### 단일 영상 스크립트 추론

```python
from src.inference import BodyFitInference

engine = BodyFitInference(exercise="squat", ckpt_path="checkpoints/cvae_squat/best.pt")
results = engine.run("my_squat.mp4")

for r in results:
    print(f"Rep {r['rep_idx']+1}: score={r['anomaly_score']:.3f}  {'이상' if r['is_anomaly'] else '정상'}")
    print(r['feedback'])
```

---

## 평가 결과

### Synthetic OOD (학습 분포 k=8σ 이탈, 200:200)

| 모델 | AUROC | 비고 |
|------|-------|------|
| CVAE (체형 조건화) | **0.9528** | 메인 모델 |
| CVAE (체형 없음) | 0.9532 | 체형 조건화 효과는 val loss로 확인 |
| cluster_cond | **0.9106** | BC-STNF 계열 최고 |
| mlp_feat | 0.6967 | ST-GCN 제거 시 저하 |
| BC-STNF | 0.5002 | ST-GCN 스무딩 + flow collapse |

### 체형 조건화 효과

| 모델 | val loss | 설명 |
|------|----------|------|
| CVAE (체형 있음) | **0.7478** | 체형 개인차 반영 → 분포 집중 |
| CVAE (체형 없음) | 0.9587 | 체형 무시 → 분포 산만 |

AUROC는 synthetic 평가에서 동일하지만, 이는 synthetic 이상이 body feature를 변경하지 않기 때문. 실제 다양한 체형 사용자에서 효과 기대.

### 실제 Tier3 (GPT-4o 라벨, 59:59 balanced)

모든 모델 AUROC ≈ 0.5. **원인**: 학습·평가 데이터 모두 YouTube 선수 영상 → 동일 분포. 파워리프팅 특유의 미세 오류가 학습 분포 내에 존재.

---

## 체형 7차원 벡터

| 인덱스 | 의미 | 핵심 종목 |
|--------|------|----------|
| b[0] | thigh/shin — 대퇴/경골 비율 | Squat, Deadlift |
| b[1] | torso/leg — 상체/하체 비율 | 전종목 |
| b[2] | arm/torso — 팔/몸통 비율 | Bench, Deadlift, OHP |
| b[3] | shoulder/hip — 어깨/엉덩이 너비 비율 | 전종목 |
| b[4] | armspan/height — 윙스팬/키 | Bench ROM |
| b[5] | pelvic_tilt — 골반 기울기 | Squat, Deadlift |
| b[6] | sym — 좌우 대칭도 | 전종목 |

모두 비율 기반 (scale-invariant) — 카메라 거리 무관.

---

## 주요 설계 결정

| 이슈 | 원인 | 해결 |
|------|------|------|
| BC-STNF OOM | ST-GCN 64ch → D=135,168 → Linear 18GB | 채널 3→8→4, mean pooling → D=132 |
| Flow collapse | L2 reg(λ=1e-2) → s→0 → near-identity | ActNorm + zero-init + s·tanh×2.0 |
| body_feature 이상치 | 정면 촬영 시 다리 투영 거리≈0 → b[0], b[6] 폭발 | `np.abs(body)>100` 필터링 |
| crawl.py TypeError | `--get-duration`이 숫자 단독 라인 출력 → int → `item["id"]` 실패 | `--print`만 사용 |
| MediaPipe `mp.solutions` | 0.10+ 이후 solutions API 제거 | Tasks API + `.task` 모델 파일 |

---

## 참고 문헌

- Hirschorn & Avidan. *Normalizing Flows for Human Pose Anomaly Detection*. ICCV 2023
- Dinh et al. *Density Estimation using Real NVP*. ICLR 2017
- Perez et al. *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018
- Yan et al. *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018
- *Anthropometry-Aware Deep Learning for Exercise Assessment*. PMC 2025

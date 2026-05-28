# BodyFit — CLAUDE.md

## 프로젝트 개요

**Body-Aware Form Assessment for Big4 Powerlifts**
체형 조건부 딥러닝 모델로 개인 맞춤형 자세 이상을 탐지하는 시스템.

- **학번/이름**: 24013840 이동현
- **과목**: 딥러닝 실습 (2026 Spring)
- **목표**: 선수·가이드 영상만으로 P(pose | body) 분포를 학습 → 사용자 자세의 이상 점수 A(x) = -log P 계산

**핵심 차별점 3가지**:
1. Body-Conditioned Flow — 체형 7차원 벡터를 조건 입력으로 P(pose|body) 학습
2. Label-free Anomaly — Positive-only 학습으로 라벨링 부담 제거, open-set 이상 탐지
3. Interpretable Feedback — log P 기반 수치 + 관절별 기여도 heatmap + Claude API 자연어 피드백

---

## 대상 운동 (Big4)

| 운동 | 분할 기준 관절 | Tier 1 목표 | rep 목표 |
|------|----------------|-------------|----------|
| Squat | 무릎각 | ~300 영상 | ~1,000 rep |
| Bench Press | 팔꿈치각 | ~300 영상 | ~1,200 rep |
| Deadlift | 고관절각 | ~250 영상 | ~800 rep |
| Overhead Press | 어깨각 | ~150 영상 | ~500 rep |

---

## 모델 스택

```
① MediaPipe BlazePose  — pose 추출 (기성, 학습 없음)
② CVAE                 — Phase 1 baseline / fallback 안전장치
③ BC-STNF              — 핵심 모델 (직접 학습)
④ Claude API           — LLM 후처리 (수치 → 자연어 피드백)
```

### BC-STNF 아키텍처 상세

```
Input:  pose P ∈ R^(64×33×3),  body b ∈ R^7
         ↓
① Body Encoder (MLP):  b → 7→32→16  →  c ∈ R^16  (ReLU + LayerNorm)
         ↓
② ST-GCN × 2 + FiLM:  관절 그래프 + 시간 conv
   feat' = feat · γ(c) + β(c)   (channel-wise scale·shift)
         ↓
③ Conditional RealNVP × 6:  가역 변환, c를 각 coupling layer 조건으로 사용
   z = f(pose; c) ~ N(0, I)
         ↓
Output: A(x) = -log P(pose|body)  +  joint attribution heatmap (T=64, J=33)
```

**손실 함수**: L = -log P(pose|body) = NLL (Maximum Likelihood)

### CVAE 아키텍처 (Fallback)

```
Encoder:  (pose, body) → μ, σ
Decoder:  (z, body) → pose_reconstruction
Score:    reconstruction error = 이상 점수
```

---

## 체형 7차원 Feature

```python
body_vec = [thigh/shin,    # b[0] 대퇴/경골 — 스쿼트·데드 자세 결정 핵심
            torso/leg,     # b[1] 상체/하체 비율
            arm/torso,     # b[2] 팔/몸통 — 벤치·데드·OHP 그립 위치
            shoulder/hip,  # b[3] 어깨비/엉덩이비 — V-taper 반영
            armspan/h,     # b[4] 윙스팬/키 — 벤치 ROM에 직접 영향
            pelvic_tilt,   # b[5] 골반 전·후방 기울기
            sym]           # b[6] 좌우 대칭도
```

모두 정적 프레임에서 측정, 비율 기반 (scale-invariant).

---

## Inference Pipeline (5단계)

```
① INPUT          영상 업로드 (mp4, 5~30초, 720p+)
② POSE           MediaPipe BlazePose → (T, 33, 4)
③ PREPROCESSING  Hip-center 정규화 + rep 분할 + 64프레임 resample → (64,33,3) + body(7,)
④ BC-STNF        log P(pose|body) 계산 + gradient → joint heatmap
⑤ OUTPUT         rep 점수 + heatmap + Claude API 자연어 피드백 2~3문장
```

### Preprocessing 세부

- visibility < 0.5 마스킹 → 1€ filter → 선형 보간
- Hip-center를 원점 (0,0)으로 이동
- 어깨너비를 1.0으로 스케일 → 카메라 거리 무관
- 종목별 기준 관절 각도 시계열로 `scipy.signal.find_peaks` → rep 경계 결정
- 각 rep을 64 frame으로 균등 resample

---

## Training Pipeline (6단계)

| 단계 | 내용 | 목표 |
|------|------|------|
| T1 · 데이터 수집 | yt-dlp 크롤링 | ~1,000 영상 |
| T2 · Keypoint 추출 | MediaPipe + rep 분할 + body feature | ~3,000 rep 쌍 |
| T3 · Phase 1 CVAE | 안전장치 baseline | MVP 확보 |
| T4 · Phase 2 BC-STNF | 핵심 모델 학습 | AUROC ≥ 0.85 |
| T5 · Threshold 튜닝 | val set 5th percentile | EER ≤ 0.15 |
| T6 · 평가 | 지표 4종 + Ablation 5종 | 보고서 완성 |

**BC-STNF 학습 설정**:
- Optimizer: AdamW, lr=3e-4, cosine schedule
- Epochs: 30, grad clip 1.0
- K-means oversampling (희귀 체형 보완)
- Batch: pose (64,33,3) + body (7,) 쌍

---

## 데이터 전략

### Tier 1 — 주 학습 데이터 (YouTube yt-dlp)
선수·가이드 영상 = Positive-only 라벨 (별도 어노테이션 불필요)
- IPF 파워리프팅 대회 공식 영상
- Squat University, Jeff Nippard, Starting Strength, Juggernaut

### Tier 2 — 보조 (공개 데이터셋)
Fit3D, InfiniteRep, MM-Fit — Tier 1 부족 시 보완

### Tier 3 — 평가셋
일반 YouTube 영상, 오류 포함 (~270 rep), LLM + 수동 라벨링

**라벨링**: Tier 3만 Claude API로 초안 → 수동 검수

---

## 평가 지표 및 목표

| Metric | 목표 | 설명 |
|--------|------|------|
| AUROC | ≥ 0.85 | rep 단위 정상/이상 분류 |
| PR-AUC | ≥ 0.75 | 정상>이상 불균형 환경 |
| EER | ≤ 0.15 | FPR = FNR 지점 |
| Per-joint IoU | ≥ 0.50 | heatmap vs 수동 라벨 일치 |
| Inference Latency | ≤ 1초/rep | CPU 기준 실시간성 |

### Ablation 5종

1. 체형 조건화 유무 (with vs without body conditioning)
2. CVAE vs BC-STNF (Flow의 exact likelihood 이점)
3. ST-GCN vs 단순 MLP
4. Cluster vs Continuous body encoding
5. 종목별 분리 모델 vs 통합 모델

---

## 디렉토리 구조

```
bodyfit/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── raw/               # yt-dlp 다운로드 원본 영상
│   ├── keypoints/         # MediaPipe 추출 결과 (.npy, shape: T×33×4)
│   ├── processed/         # 전처리 완료 (pose: 64×33×3, body: 7) 쌍
│   └── test/              # Tier 3 평가셋 (오류 라벨 포함)
├── src/
│   ├── data/              # 크롤링·전처리 모듈
│   │   ├── crawl.py       # yt-dlp 크롤링 스크립트
│   │   ├── extract_keypoints.py   # MediaPipe 배치 처리
│   │   ├── preprocess.py  # 정규화·rep 분할·resample
│   │   └── body_feature.py        # 체형 7차원 벡터 추출
│   ├── models/            # 모델 정의
│   │   ├── body_encoder.py        # MLP: 7→32→16
│   │   ├── st_gcn.py      # ST-GCN + FiLM conditioning
│   │   ├── realnvp.py     # Conditional RealNVP coupling layers
│   │   ├── bc_stnf.py     # BC-STNF 통합 모델
│   │   └── cvae.py        # CVAE baseline/fallback
│   ├── training/          # 학습 스크립트
│   │   ├── train_cvae.py
│   │   └── train_bc_stnf.py
│   └── evaluation/        # 평가·ablation
│       ├── metrics.py     # AUROC, PR-AUC, EER, Per-joint IoU
│       ├── ablation.py    # Ablation 5종 실행
│       └── llm_feedback.py        # Claude API 자연어 피드백
├── notebooks/             # 실험·시각화 (EDA, 결과 분석)
├── checkpoints/           # 모델 저장 (.pt)
└── results/               # 평가 결과·시각화 (.json, .png)
```

---

## Claude에 대한 지침

### 우선순위 원칙

1. **계획서 범위 엄수** — 빅4 전종목, BC-STNF, Ablation 5종 모두 계획서대로 구현. 범위 축소는 감점 사유이므로 절대 제안하지 말 것.
2. **Positive-only 학습** — 오류 라벨 생성하지 말 것. 학습 데이터는 무조건 정상으로 처리.
3. **체형 조건화** — FiLM conditioning이 핵심 novelty. 단순 concat으로 대체하지 말 것.
4. **exact log-likelihood** — Flow 모델의 핵심. reconstruction error(VAE 방식)로 대체하면 novelty 소멸.

### 코드 작성 지침

- PyTorch 2.x 기준, `torch.no_grad()`로 inference 모드 관리
- body feature는 항상 7차원 (thigh/shin, torso/leg, arm/torso, shoulder/hip, armspan/h, pelvic_tilt, sym)
- pose 입력은 항상 hip-center 정규화 + 어깨너비 1.0 스케일 후 사용
- rep 단위로 처리 (영상 단위 아님) — 각 rep을 64 frame으로 resample
- MediaPipe keypoint index는 COCO 17+ 확장 (33 keypoints)

### 핵심 수식

```python
# 이상 점수
anomaly_score = -log_prob  # A(x) = -log P(pose|body)

# FiLM conditioning
gamma, beta = mlp(c)       # c: body embedding (16차원)
feat = feat * gamma + beta  # channel-wise scale·shift

# RealNVP coupling
x_a = x_a                  # 그대로 통과
x_b = x_b * exp(s(x_a, c)) + t(x_a, c)  # c로 조건화
log_det = sum(s)            # log-determinant
```

### 개발 순서

1. 데이터 수집 스크립트 (yt-dlp)
2. MediaPipe 전처리 파이프라인 + body feature 추출
3. CVAE baseline
4. BC-STNF 구현 + 학습
5. Ablation 5종 + 평가
6. LLM 통합 + FastAPI

---

## 구현 현황 (2026-05-28)

### 완료

#### 환경
- conda 환경 `bodyfit` (Python 3.11, PyTorch 2.12 + MPS) — 맥북 로컬
- Backend.AI 서버: 24.09 이미지 (Python 3.10, CUDA 12.6), CPU 48코어, RAM 64GiB
- GitHub: https://github.com/Hyun0930/Bodyfit.git (최신 커밋 0ec1ea3)

#### Phase 1 — 데이터 파이프라인 (`src/data/`) ✅ 완료

| 파일 | 역할 | 상태 |
|------|------|------|
| `crawl.py` | yt-dlp 종목별 크롤링, metadata.csv, 중복 방지 | ✅ 완료 |
| `extract_keypoints.py` | MediaPipe Tasks API (0.10+), 1€ filter, 멀티프로세싱 `--workers` | ✅ 완료 |
| `preprocess.py` | hip-center 정규화, find_peaks rep 분할, 64프레임 resample | ✅ 완료 |
| `body_feature.py` | scale-invariant 체형 7차원 벡터 추출 | ✅ 완료 |
| `__init__.py` | `BodyFitDataset` (torch Dataset), train/val split | ✅ 완료 |

**파이프라인 검증 완료**: (15821, 33, 4) → 526 reps → pose(64,33,3) + body(7,)

#### 서버 스크립트
- `setup_server.sh` — 환경 세팅 (CUDA 자동 감지, MediaPipe 모델 다운, 디렉토리 생성)
- `run_pipeline.sh` — crawl → extract → preprocess 전체 자동화, nohup 지원

#### Phase 2 — 모델 (`src/models/`) ✅ 완료

| 파일 | 역할 | 상태 |
|------|------|------|
| `cvae.py` | CVAE baseline (Encoder/Decoder/anomaly_score/compute_loss) | ✅ 완료 |
| `body_encoder.py` | MLP 7→32→16, LayerNorm | ✅ 완료 |
| `st_gcn.py` | ST-GCN × 2 (3→8→4채널) + FiLM conditioning, BlazePose 인접행렬 | ✅ 완료 |
| `realnvp.py` | Conditional RealNVP × 6 coupling layers | ✅ 완료 |
| `bc_stnf.py` | BC-STNF 통합 모델, joint_attribution heatmap | ✅ 완료 |

**BC-STNF 설계 결정**:
- ST-GCN 출력 채널: 3→8→4 (D=8,448) — 원래 64채널은 Linear weight 18GB로 OOM
- Flow 입력: ST-GCN 출력 전체 flatten (mean pooling 없음) — P(pose|body) 직접 모델링
- 파라미터 총 321M (~1.2GB weights), 맥북 M3 학습 가능
- 채널 수는 하이퍼파라미터 — 데이터 학습 후 성능 보며 조정 가능

#### Phase 3 — 학습 스크립트 (`src/training/`) ✅ 완료

| 파일 | 역할 | 상태 |
|------|------|------|
| `train_cvae.py` | AdamW + CosineAnnealing, best checkpoint 저장, 학습 곡선 | ✅ 완료 |
| `train_bc_stnf.py` | AdamW + CosineAnnealing + K-means oversampling, grad clip 1.0 | ✅ 완료 |

### 진행 중

- **Backend.AI 서버에서 데이터 수집 실행 중** (2026-05-28 재시작)
  - 44 workers, CPU 48코어, nohup 백그라운드
  - 로그: `/home/work/body_fit/pipeline.log`
  - squat 크롤링 9/300 확인 → 정상 동작 중

### 다음 단계

- [ ] Phase 4: 평가 스크립트 (`src/evaluation/metrics.py`, `ablation.py`, `llm_feedback.py`)
- [ ] Phase 5: 데이터 수집 완료 후 실제 학습 실행
- [ ] Phase 6: Ablation 5종 + 최종 평가

### 주요 기술 결정 및 트러블슈팅

| 이슈 | 원인 | 해결 |
|------|------|------|
| `mp.solutions` AttributeError | MediaPipe 0.10+에서 solutions API 제거 | Tasks API + pose_landmarker_heavy.task 사용 |
| 맥북 데이터 처리 9.5일 | MediaPipe CPU 처리 속도 한계 | Backend.AI 서버 44 worker 멀티프로세싱 |
| MediaPipe CUDA 미지원 | Linux에서 CUDA 백엔드 없음 | CPU 멀티프로세싱으로 대체 |
| vfolder/코드 경로 분리 | Backend.AI ephemeral 컨테이너 | `BODYFIT_DATA` 환경변수로 데이터 경로 분리 |
| `crawl.py` TypeError: 'int' object is not subscriptable | `--get-id/title/duration`과 `--print` 동시 사용 시 yt-dlp가 duration을 별도 숫자 라인으로 출력 → `json.loads("120")` = int → `item["id"]` 실패 | `--get-*` 플래그 전부 제거, `--print`만 사용 (commit f82c5a8) |
| `crawl.py` 재실행 시 max_videos 초과 | `downloaded = 0`으로 시작해 기존 파일을 카운트에 미포함 → 재검색 결과가 안 겹치면 기존+300개 다운 | `downloaded = len(existing)`으로 초기화 |
| BC-STNF OOM (맥북 스왑 38GB) | ST-GCN 64채널 출력 flatten → D=135,168 → Linear weight 18GB | ST-GCN 채널 3→8→4로 축소, D=8,448로 tractable |
| BC-STNF view 오류 | block1 통과 후 non-contiguous 텐서에 `.view()` 실패 | `.reshape()`으로 교체 |

### 참고 문헌

- [STG-NF] Hirschorn & Avidan. Normalizing Flows for Human Pose Anomaly Detection. ICCV 2023
- [RealNVP] Dinh et al. Density Estimation using Real NVP. ICLR 2017
- [FiLM] Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018
- [ST-GCN] Yan et al. Spatial Temporal Graph Convolutional Networks. AAAI 2018
- [ADA] Anthropometry-Aware Deep Learning for Exercise Assessment. PMC 2025

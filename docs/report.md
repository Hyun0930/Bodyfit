# BodyFit: 체형 조건부 딥러닝 기반 빅4 파워리프팅 자세 이상 탐지

> 세종대학교 인공지능데이터사이언스학과  
> 딥러닝 실습 기말 프로젝트  
> 학번: 24013840 | 이름: 이동현  
> GitHub: https://github.com/Hyun0930/Bodyfit.git

---

## 목차

1. 프로젝트 개요 및 동기
2. 관련 연구
3. 프로젝트 진행 과정
4. 데이터 수집 및 구성
5. 전처리 파이프라인
6. 모델 설계
7. 학습
8. Ablation 실험
9. 평가 결과
10. 추론 파이프라인 및 배포
11. 트러블슈팅
12. 결론 및 한계점
13. 참고 문헌

---

## 1. 프로젝트 개요 및 동기

### 1.1 문제 정의

파워리프팅의 빅4 종목(스쿼트, 벤치프레스, 데드리프트, 오버헤드프레스)은 잘못된 자세로 수행할 경우 허리, 무릎, 어깨 등에 심각한 부상 위험이 따른다. 이를 방지하기 위한 자세 평가 AI 시스템들이 연구되고 있으나, 기존 접근법은 모든 사용자에게 동일한 기준을 적용한다는 근본적인 한계를 가진다.

파워리프팅에서 올바른 자세의 기준은 개인의 체형에 따라 달라진다. 예를 들어:
- **대퇴골이 긴 사람**은 스쿼트 시 무게 중심을 유지하기 위해 몸을 더 앞으로 기울여야 한다. 이는 허리가 약한 자세처럼 보이지만 그 사람에게는 올바른 형태다.
- **팔이 긴 사람**은 벤치프레스에서 ROM(운동 범위)이 커지고 자연스럽게 그립 위치가 달라진다.
- **골반 전방 기울기(anterior pelvic tilt)가 큰 사람**은 데드리프트 시 허리 중립 유지 방법이 달라진다.

따라서 "체형을 무시한 자세 평가"는 개인에게 맞지 않는 피드백을 줄 수 있고, 오히려 부상을 유발할 위험이 있다.

### 1.2 제안 아이디어

본 프로젝트는 이 문제를 해결하기 위해 **체형 조건부(body-conditioned) 이상 탐지** 접근법을 제안한다. 핵심 아이디어는 다음과 같다.

- 선수·유튜브 가이드 영상만을 Positive 데이터로 사용하여 정상 자세 분포로 학습한다.
- 이를 통해 정상/비정상을 검수 및 구분하고 라벨링하는 비용 부담을 제거한다. (Positive-only, label-free 학습)
- 체형을 7차원 벡터로 수치화하여 모델의 조건 입력으로 활용한다.
- **P(pose | body)** 분포를 학습하여 이상 점수 **A(x) = −log P(pose | body)** 를 계산한다.

이 방식은 STG-NF(ICCV 2023)의 보행 이상 탐지 접근법에서 착안하여, 파워리프팅 도메인으로 확장하고 체형 조건화를 추가한 것이다.

### 1.3 핵심 차별점

기존 자세 평가 시스템과 본 프로젝트의 차별점은 세 가지다.

**① 체형 조건부 학습 (Body-Conditioned)**
체형 7차원 벡터를 모델의 조건 입력으로 사용하여 P(pose | body)를 학습한다. 동일한 자세라도 체형에 따라 정상 여부가 달라지는 파워리프팅의 특성을 반영한 핵심 설계다.

**② Label-free 이상 탐지**
선수·가이드 영상만으로 정상 분포를 학습하는 Positive-only 방식을 채택하였다. 이상 자세에 대한 별도 라벨링 없이 학습이 가능하여 데이터 수집 비용을 크게 절감한다.

**③ 해석 가능한 피드백**
이상 점수 수치 외에 관절별 기여도 heatmap과 GPT-4o 기반 한국어 자연어 피드백을 함께 제공하여, 사용자가 어느 관절에서 문제가 발생했는지 직관적으로 파악할 수 있다.

---

## 2. 관련 연구

### 2.1 STG-NF (Hirschorn & Avidan, ICCV 2023)

*Normalizing Flows for Human Pose Anomaly Detection*

ST-GCN으로 골격 포즈의 시공간 특징을 추출한 뒤 Normalizing Flow로 정상 분포를 학습하는 프레임워크다. 보행(gait) 영역에서 Positive-only 학습만으로 높은 이상 탐지 성능을 달성하였다. ST-GCN이 관절 그래프 구조를 통해 보행의 시공간 패턴을 효과적으로 모델링하고, 정규화 흐름이 exact log-likelihood를 제공한다는 점이 핵심이다.

**본 프로젝트와의 관계**: BC-STNF 설계의 직접적인 출발점. 체형(body feature) conditioning 없이 포즈만 학습하는 한계가 있어, FiLM 기반 체형 조건화를 추가하는 것이 본 프로젝트의 출발점이다. 다만 보행 이상(쓰러짐, 비정상 행동)과 달리 파워리프팅 이상은 정상-이상 경계가 모호하여 동일 접근법의 적용에 한계가 있음을 실험을 통해 확인하였다.

### 2.2 RealNVP (Dinh et al., ICLR 2017)

*Density Estimation using Real NVP*

Coupling layer를 기반으로 한 정규화 흐름(Normalizing Flow) 모델이다. 입력 x를 두 파트(x_a, x_b)로 나눠 다음과 같이 변환한다.

```
x_a' = x_a                          (identity pass)
x_b' = x_b ⊙ exp(s(x_a)) + t(x_a)  (affine transform)
log|det J| = sum(s(x_a))            (log-determinant)
```

이 구조는 변환이 역방향으로 쉽게 계산되고(`x_b = (x_b' - t) / exp(s)`), Jacobian의 log-determinant가 대각행렬로 단순 합산되어 exact log-likelihood 계산이 가능하다.

**본 프로젝트와의 관계**: BC-STNF의 Flow 모듈로 채택. 조건 벡터 c(체형 임베딩)를 `s(x_a, c)`, `t(x_a, c)`에 추가 입력으로 주입하여 Conditional RealNVP로 확장하였다.

### 2.3 FiLM (Perez et al., AAAI 2018)

*FiLM: Visual Reasoning with a General Conditioning Layer*

Feature-wise Linear Modulation. 조건 정보 c를 받아 신경망 중간 특징(feature)에 channel-wise scale·shift를 가하는 방법이다.

```
feat' = γ(c) ⊙ feat + β(c)
```

γ(c)와 β(c)는 조건 벡터 c로부터 학습된 MLP가 생성한다. 원 논문은 시각 질문 응답(VQA) 태스크에서 언어 조건을 비전 특징에 주입하는 데 사용하였으나, 이후 다양한 조건부 생성 및 인식 모델에서 광범위하게 채택되었다.

**본 프로젝트와의 관계**: ST-GCN 출력 특징에 체형 임베딩 c를 주입하는 방법으로 FiLM을 채택하였다. 단순 concat 대비 채널별로 체형 정보를 scale·shift하여 각 특징 차원이 체형에 맞게 선택적으로 강조·억제될 수 있다는 장점이 있다.

### 2.4 ST-GCN (Yan et al., AAAI 2018)

*Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*

골격 기반 행동 인식을 위한 그래프 신경망 구조다. 사람의 관절을 그래프 노드로, 관절 간 연결(뼈)을 엣지로 모델링하여 공간적 관계를 학습(GCN)하고, 시간 축 Conv를 통해 시간적 패턴도 함께 학습(TCN)한다.

```
공간(GCN): h = D^(-1/2) A D^(-1/2) X W  (대칭 정규화 인접행렬)
시간(TCN): Conv1d along T axis
```

**본 프로젝트와의 관계**: 파워리프팅 운동 자세의 시공간 패턴을 추출하는 특징 추출기로 사용하였다. 그러나 ST-GCN이 인접 관절 신호를 평균화하는 스무딩 효과가 이상 탐지에서 관절 변형 신호를 희석하는 문제를 발견하였다.

### 2.5 ADA (PMC 2025)

*Anthropometry-Aware Deep Learning for Exercise Assessment*

체형 측정값(anthropometry)을 딥러닝 운동 평가 모델에 통합하면 개인화된 평가 품질이 향상된다는 근거를 제시한 연구다. 팔·다리 비율 등 체형 정보가 운동 자세 판단의 유효한 맥락 정보임을 실험적으로 보였다.

**본 프로젝트와의 관계**: 체형 7차원 벡터를 조건 입력으로 사용하는 본 프로젝트의 핵심 가정(체형 → 자세 기준 변화)의 이론적 근거로 활용하였다.

---

## 3. 프로젝트 진행 과정

본 프로젝트는 약 6주에 걸쳐 단계별로 진행하였다.

### 1주차 — 프로젝트 기획 및 환경 구성

- 빅4 파워리프팅 자세 이상 탐지라는 주제 확정
- 관련 연구(STG-NF, RealNVP, FiLM, ST-GCN) 조사 및 분석
- 체형 조건화 아이디어 구체화 (BC-STNF 구조 설계)
- conda 환경 구성, MediaPipe Tasks API 설치 및 테스트
- 프로젝트 디렉토리 구조 설계

### 2주차 — 데이터 수집 파이프라인 구현 및 크롤링

- `src/data/crawl.py` 구현: yt-dlp 기반 YouTube 크롤링, 종목별 검색어 설정, 중복 방지, metadata.csv 저장
- `src/data/extract_keypoints.py` 구현: MediaPipe BlazePose Tasks API 기반 배치 처리, 1€ 필터 노이즈 제거, 멀티프로세싱(`--workers`) 지원
- 빅4 4종목 YouTube 영상 수집 (각 150~300개 영상)
- 크롤링 관련 트러블슈팅: yt-dlp `--get-duration` 충돌, 재실행 시 max_videos 초과 등

### 3주차 — 전처리 파이프라인 및 체형 벡터 추출 구현

- `src/data/preprocess.py` 구현: hip-center 정규화, 어깨너비 스케일, `scipy.signal.find_peaks` 기반 rep 분할, 64프레임 resample
- `src/data/body_feature.py` 구현: 정적 프레임에서 체형 7차원 비율 벡터 추출
- `src/data/__init__.py`: `BodyFitDataset` (PyTorch Dataset), train/val split
- 체형 벡터 이상치 발견 및 필터링 (`abs(body) > 100`)
- 전체 90,514 rep 전처리 완료

### 4주차 — 모델 구현

- `src/models/body_encoder.py`: MLP 7→32→16 (Body Encoder)
- `src/models/st_gcn.py`: BlazePose 인접행렬 기반 STGCNBlock (GCN + TCN + FiLM)
- `src/models/realnvp.py`: Conditional RealNVP × 6 coupling layers, ActNorm, zero-init
- `src/models/bc_stnf.py`: BC-STNF 통합 모델 (joint attribution heatmap 포함)
- `src/models/cvae.py`: CVAE baseline (Encoder/Decoder/anomaly_score/joint_attribution)
- BC-STNF 설계 과정에서 D=135,168 OOM 문제 발견 → Temporal Mean Pooling으로 D=132 축소

### 5주차 — 학습 및 평가셋 구축

- `src/training/train_cvae.py`, `train_bc_stnf.py`, `train_ablation.py` 구현 및 실행
- CVAE, BC-STNF, Ablation 5종(no_cond, mlp_feat, cluster_cond, per_exercise, no_body_cvae) 전체 학습
- Tier2 평가셋 구축: 일반 YouTube 영상 수집 (`src/data/tier3_dataset.py`)
- `src/data/label_tier3.py`: GPT-4o API를 활용한 자동 라벨링 + 수동 검수 → 365 rep 라벨 완성
- `src/evaluation/dist_synth_eval.py`: 학습 분포 기반 Synthetic OOD 생성 및 평가 구현

### 6주차 — 평가·분석 및 배포

- `src/evaluation/metrics.py`: AUROC, PR-AUC, EER, Per-joint IoU 구현
- `src/evaluation/ablation.py`: Ablation 5종 실험 실행 및 결과 분석
- `src/evaluation/score_dist.py`: 점수 분포 분석
- `src/inference.py`: 단일 영상 end-to-end 추론 파이프라인 구현
- `app.py`: FastAPI 서버 구현
- `static/index.html`: 웹 프론트엔드 구현
- `Dockerfile`, `docker-compose.yml`: Docker 배포 환경 구성
- `scripts/calc_threshold.py`: val set 95th percentile threshold 계산 및 체크포인트 저장

---

## 4. 데이터 수집 및 구성

### 4.1 데이터 계층 구조

데이터를 역할에 따라 두 계층으로 구성하였다.

- **Tier 1**: 학습 데이터 — 선수·가이드 YouTube 영상 (Positive-only)
- **Tier 2**: 평가 데이터 — 오류 동작을 포함한 일반 YouTube 영상 (정상/이상 라벨)

*(공개 데이터셋(Fit3D, InfiniteRep, MM-Fit) 보완은 Tier 1 데이터가 충분히 수집되어 사용하지 않았다.)*

### 4.2 Tier 1 — 학습 데이터

`yt-dlp`를 활용한 크롤링 스크립트(`src/data/crawl.py`)로 종목별 YouTube 영상을 자동 수집하였다. IPF 파워리프팅 대회 공식 영상, Squat University, Jeff Nippard, Starting Strength, Juggernaut 채널을 주요 출처로 사용하였다. 선수·코치가 촬영한 영상만 수집하므로 별도의 정상/이상 라벨링 없이 전체를 Positive(정상) 데이터로 사용한다.

| 종목 | 수집 영상 수 | 필터링 후 학습 reps |
|------|-------------|-------------------|
| Squat | 약 300개 | 11,250 |
| Bench Press | 약 300개 | 38,489 |
| Deadlift | 약 250개 | 28,078 |
| Overhead Press | 약 150개 | 12,697 |
| **합계** | **약 1,000개** | **90,514** |

*(필터링: 체형 벡터 이상치 `abs(body) > 100` 제거, MediaPipe 추출 실패 영상 제외)*

### 4.3 Tier 2 — 평가 데이터

일반 YouTube 영상에서 오류 동작을 포함한 영상을 수집하였다. 수집 검색어는 종목별 오류 관련 키워드를 사용하였다 (예: "squat form check beginner", "squat bad form correction", "deadlift back rounding fix" 등, 상세 쿼리는 `src/data/crawl.py`의 `TIER3_QUERIES` 참고). GPT-4o API로 종목별 기준에 따라 초안 라벨링(0=정상, 1=이상)을 생성하고, 수동 검수를 거쳐 최종 라벨을 확정하였다. 라벨링 기준은 자세 평가 기준(무릎 cave-in, 허리 라운딩, 엉덩이 상승 등)을 종목별로 GPT-4o 시스템 프롬프트에 명시하였다.

| 종목 | 정상 reps | 이상 reps | 총 reps |
|------|----------|----------|--------|
| Squat | 25 | 386 | 411 |
| Bench Press | 16 | 524 | 540 |
| Deadlift | 9 | 328 | 337 |
| OHP | 9 | 322 | 331 |
| **합계** | **59** | **1,560** | **1,619** |

*(이상 rep이 압도적으로 많아 평가 시 balanced 59:59 샘플링 적용)*

### 4.4 체형 7차원 벡터

포즈 영상의 대표 프레임(정지 자세 구간)에서 MediaPipe keypoint 간 유클리드 거리 비율로 추출한다. 모두 비율 기반(scale-invariant)이므로 카메라 거리와 무관하다.

```python
body = [
    thigh / shin,          # b[0]: 대퇴/경골 비율 — 스쿼트·데드 자세 결정
    torso / leg,           # b[1]: 상체/하체 비율
    arm / torso,           # b[2]: 팔/몸통 비율 — 벤치·OHP 그립 위치
    shoulder_w / hip_w,    # b[3]: 어깨/엉덩이 너비 비율
    armspan / height,      # b[4]: 윙스팬/키 — 벤치 ROM에 직접 영향
    pelvic_tilt,           # b[5]: 골반 전·후방 기울기
    symmetry,              # b[6]: 좌우 대칭도
]
```

---

## 5. 전처리 파이프라인

전처리는 원본 영상에서 모델 입력 형식인 `(64, 33, 3)` + `body (7,)` 쌍을 생성하는 과정이다.

### 5.1 Keypoint 추출 (MediaPipe BlazePose)

MediaPipe BlazePose Tasks API를 사용하여 영상의 각 프레임에서 33개 keypoint를 추출한다. 각 keypoint는 `(x, y, z, visibility)` 4차원으로 구성되며, 출력은 `(T, 33, 4)` 형태다. (`T`: 총 프레임 수)

원본 영상(mp4) → MediaPipe BlazePose → 출력: **(T, 33, 4)**

각 keypoint의 4개 채널:
- **x, y**: 프레임 내 정규화 좌표 [0, 1]
- **z**: 엉덩이 중심 기준 깊이
- **visibility**: 관절 가시성 [0, 1]

- **MediaPipe 버전 이슈**: 0.10+ 이후 `mp.solutions` API가 제거되어 Tasks API + `.task` 모델 파일 방식으로 전면 교체하였다. `pose_landmarker_heavy.task` 모델(가장 정확도 높은 variant)을 사용한다.

### 5.2 노이즈 제거 및 결측 처리

- **1€ 필터**: 각 관절 좌표에 적용. 빠른 움직임은 적게 스무딩, 느린 움직임은 강하게 스무딩하여 움직임 반응성을 유지하면서 노이즈를 제거한다.
- **visibility 마스킹**: `visibility < 0.5`인 관절을 결측으로 처리하고 인접 프레임 값으로 선형 보간한다. (예: 데드리프트에서 카메라에 바가 가려져 손목이 보이지 않는 경우)

### 5.3 정규화

```
1. Hip-center 정규화
   hip_center = (left_hip + right_hip) / 2
   keypoints -= hip_center          # 엉덩이 중심을 원점 (0, 0, 0)으로 이동

2. 어깨너비 스케일
   shoulder_width = |left_shoulder - right_shoulder|
   keypoints /= shoulder_width      # 어깨너비를 1.0으로 정규화 → 카메라 거리 무관
```

이 두 정규화를 통해 촬영 거리, 카메라 높이, 피사체 위치가 달라도 동일한 체형의 동일한 자세는 같은 입력 벡터로 표현된다.

### 5.4 Rep 분할

파워리프팅은 복수의 rep(반복 동작)으로 구성되므로, 영상에서 각 rep의 경계를 자동으로 탐지해야 한다.

```python
종목별 기준 관절 각도:
  squat     → 무릎각 (knee angle)
  bench     → 팔꿈치각 (elbow angle)
  deadlift  → 고관절각 (hip angle)
  ohp       → 어깨각 (shoulder angle)

scipy.signal.find_peaks(angle_series, ...)  →  rep 경계 인덱스 탐지
→  각 rep을 개별 배열로 분리
```

각 rep은 기준 관절 각도가 최솟값(bottom position)을 기준으로 전후 구간을 잘라낸다.

### 5.5 64프레임 Resample

각 rep마다 길이가 다르기 때문에(빠른 rep은 30프레임, 느린 rep은 120프레임 등), 균일한 모델 입력을 위해 64프레임으로 균등 resample한다.

```python
indices = np.linspace(0, len(rep) - 1, 64)
pose_64 = rep[indices.astype(int)]  # (64, 33, 3)
```

### 5.6 체형 벡터 추출

rep 분할과 별개로, 영상 전체에서 가장 안정적인 정지 구간(standing position)을 찾아 체형 7차원 벡터를 한 번 추출한다. 이 값은 해당 영상의 모든 rep에 동일하게 사용된다.

```
최종 출력 per rep:
  pose: (64, 33, 3)  — 정규화된 관절 좌표 시퀀스
  body: (7,)         — 해당 인물의 체형 비율 벡터
```

---

## 6. 모델 설계

### 6.1 CVAE (Conditional Variational Autoencoder) — Baseline

CVAE는 조건부 변분 오토인코더로, 정상 포즈의 잠재 분포를 학습하고 재구성 오류를 이상 점수로 사용한다.

#### 구조

```
Encoder:
  입력: [pose_flat (6336) + body (7)] = 6343차원
  6343 → Linear(1024) → ReLU → LayerNorm
       → Linear(512)  → ReLU → LayerNorm
       → Linear(256)  → [μ (128), log_σ² (128)]

Decoder:
  입력: [z (128) + body (7)] = 135차원
  135 → Linear(512)  → ReLU → LayerNorm
      → Linear(1024) → ReLU → LayerNorm
      → Linear(6336) → pose 재구성 (64×33×3)
```

pose (64, 33, 3)은 6336차원으로 flatten하여 body (7,)와 concat한 뒤 Encoder에 입력한다. Encoder와 Decoder 모두에 체형 벡터 body가 입력되어, 체형에 맞는 잠재 분포와 재구성이 이루어진다.

#### 학습 목적 함수 (ELBO)

VAE는 Evidence Lower BOund(ELBO)를 최대화하는 방향으로 학습한다. 이를 손실 함수로 표현하면:

```
L = L_recon + β × L_KL

L_recon = MSE(pose_recon, pose_flat)           # 재구성 오류
L_KL    = -0.5 × mean(1 + log_σ² - μ² - σ²)  # KL divergence vs N(0,I)
β = 1.0  (표준 VAE 설정)
```

**L_recon**: Decoder가 얼마나 원래 포즈를 잘 복원하는지 측정한다. 정상 포즈는 잠재 공간에서 잘 표현되어 재구성 오류가 낮고, 이상 포즈는 잠재 공간에서 표현하기 어려워 재구성 오류가 높다.

**L_KL**: 잠재 변수 z의 분포가 표준 정규분포 N(0, I)에 가깝도록 제약한다. 이를 통해 잠재 공간이 연속적이고 규칙적으로 구성된다.

#### 이상 점수 계산

```python
def anomaly_score(pose, body):
    mu, _ = encode(pose_flat, body)
    pose_recon = decode(mu, body)          # μ 사용 (deterministic)
    return MSE(pose_recon, pose_flat)      # per sample 평균
```

추론 시에는 확률적 샘플링 없이 μ를 직접 사용하여 결정론적 이상 점수를 계산한다. 점수가 높을수록 재구성이 잘 안 되는 자세, 즉 이상 자세를 의미한다.

#### Joint Attribution Heatmap

```python
def joint_attribution(pose, body):
    mu, _ = encode(pose_flat, body)
    pose_recon = decode(mu, body).view(B, 64, 33, 3)
    return (pose_recon - pose).pow(2).sum(dim=-1).sqrt()  # (B, 64, 33)
```

관절별(33개) 프레임별(64개) 재구성 오류의 L2 norm을 계산하여, 어느 관절이 어느 시점에 가장 잘못 재구성됐는지를 heatmap으로 표현한다. 이를 통해 이상 점수의 원인이 되는 관절을 해석적으로 파악할 수 있다.

---

### 6.2 BC-STNF (Body-Conditioned ST-GCN Normalizing Flow) — 핵심 제안 모델

BC-STNF는 STG-NF(ICCV 2023)를 기반으로 체형 FiLM conditioning을 추가한 모델이다. 포즈의 시공간 구조를 ST-GCN으로 포착하고, Normalizing Flow로 exact log-likelihood를 계산한다.

#### 전체 구조

```
입력: pose (B, 64, 33, 3)  +  body (B, 7)
              │                      │
              │               ① Body Encoder
              │               MLP: 7→32→16
              │               ReLU + LayerNorm
              │                      │
              │                      ↓  c ∈ R^16 (체형 임베딩)
              │               ┌──────┘
              ↓               ↓
        ② ST-GCN × 2 + FiLM
           STGCNBlock(3→8) → feat1 (B,64,33,8)
           FiLM: feat1' = feat1 · γ(c) + β(c)
           STGCNBlock(8→4) → feat2 (B,64,33,4)
           FiLM: feat2' = feat2 · γ(c) + β(c)
              │
              ↓
        ③ Temporal Mean Pooling
           mean(dim=T) → (B, 33, 4) → flatten → D=132
              │
              ↓
        ④ Conditional RealNVP × 6
           z = f(feat; c) ~ N(0, I)
           log P = log p(z) + Σ log|det J_i|
              │
              ↓
        이상 점수: A(x) = -log P(pose | body)
```

#### ① Body Encoder

체형 7차원 수치를 16차원 조건 벡터 c로 압축하는 MLP다.

```
7 → Linear(32) → ReLU → LayerNorm
  → Linear(16)           → c ∈ R^16
```

이 c가 이후 ST-GCN FiLM과 RealNVP coupling layer 양쪽에 조건으로 주입된다.

#### ② ST-GCN + FiLM

ST-GCN은 관절 그래프와 시간 축을 동시에 처리하는 신경망이다.

**GCN (Spatial)**: 대칭 정규화 인접행렬 A를 이용해 각 관절이 인접 관절의 정보를 집계한다.
```
A_norm = D^(-1/2) A D^(-1/2)   (자기 자신 포함, self-loop)
h = A_norm × X × W              (관절 간 메시지 전달)
```
BlazePose의 33개 keypoint 해부학적 연결(어깨-팔꿈치-손목, 엉덩이-무릎-발목 등)을 엣지로 정의하였다.

**TCN (Temporal)**: 시간 축 1D Conv로 rep 전체에서 동작 패턴의 시간적 변화를 학습한다.
```
Conv1d(kernel=9, padding=4) + BatchNorm1d
```

**FiLM**: 각 STGCNBlock 후에 체형 임베딩 c를 특징에 주입한다.
```python
gamma, beta = Linear(c_dim, feat_dim*2)(c).chunk(2)
feat' = feat * gamma[:, None, None, :] + beta[:, None, None, :]
```
channel-wise scale·shift로 각 채널이 체형에 따라 선택적으로 강조·억제된다.

#### ③ Temporal Mean Pooling

ST-GCN 출력 `(B, 64, 33, 4)`에서 시간 축(T=64)을 평균 pooling하여 `(B, 33, 4) → flatten → D=132`로 축소한다.

초기 설계에서는 전체 flatten(D=64×33×4=8,448)을 시도하였으나 Flow collapse가 발생하였다. 이는 고차원에서 log_det gradient가 log_prior gradient를 압도하여 trivial solution으로 수렴하기 때문이다. Temporal Mean Pooling으로 D=132로 축소함으로써 안정적인 학습을 달성하였다.

#### ④ Conditional RealNVP

D=132 벡터를 6개 coupling layer로 통과시켜 정규 분포 z ~ N(0, I)로 변환한다.

```
각 coupling layer:
  x → [x_a, x_b] 분할 (홀짝 번갈아)
  x_a' = x_a
  x_b' = x_b ⊙ exp(tanh(s(x_a, c)) × 2.0) + t(x_a, c)
  log|det J| = sum(tanh(s) × 2.0)

s, t: MLP(x_a ⊕ c) → [s, t]  (c: 체형 임베딩)
```

`tanh(s) × 2.0` 클리핑으로 스케일 파라미터 s의 폭발을 방지한다. 또한 각 coupling layer의 s, t MLP 마지막 레이어를 zero-init하여 학습 초기에 near-identity 변환에서 시작한다. ActNorm을 각 coupling layer 앞에 적용하여 feature 스케일을 자동 정규화한다.

#### 손실 함수 (NLL)

```
L = -log P(pose | body)
  = -log p(z) - Σ log|det J_i|
  = 0.5 × ||z||² + 0.5 × D × log(2π) - Σ log_det_i
```

Normalizing Flow는 exact log-likelihood를 직접 최대화하는 방향으로 학습한다. CVAE의 ELBO(하한 근사)와 달리 정확한 log-likelihood를 얻을 수 있다는 것이 이론적 장점이다.

---

## 7. 학습

### 7.1 CVAE 학습 설정

| 설정 | 값 |
|------|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 50 |
| Batch Size | 256 |
| 손실 함수 | MSE(재구성) + KL divergence (β=1.0) |

val loss 곡선을 모니터링하여 수렴이 확인된 epoch에서 best checkpoint를 저장하였다.

**결과**:
- CVAE (체형 있음): 최종 val loss = **0.7478**
- CVAE (체형 없음, NoBodyCVAE): 최종 val loss = **0.9587**

체형 정보를 조건으로 주입한 CVAE가 체형 없는 버전 대비 val loss가 낮다. 이는 체형 정보가 포즈 분포를 더 타이트하게 학습하는 데 기여함을 의미한다. 체형이 같은 사람들의 정상 포즈는 서로 더 유사하므로, 체형 조건화가 정상 분포를 더 집중적으로 학습할 수 있게 해준다.

### 7.2 BC-STNF 학습 설정

| 설정 | 값 |
|------|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 30 |
| Batch Size | 128 |
| Gradient Clip | 1.0 |
| s에 대한 L2 정규화 | λ=1e-2 |
| 오버샘플링 | K-means (희귀 체형 보완) |

**K-means 오버샘플링**: 체형 분포가 불균형하여(예: 팔이 매우 긴 사람이 적음) 희귀 체형의 rep을 오버샘플링하였다. 학습 데이터를 체형 벡터 기준으로 K-means 클러스터링하고, 소수 클러스터의 샘플을 복제하여 균형을 맞췄다.

**결과**: Epoch 1 val NLL = -513 → Epoch 30 val NLL = **-1443** (단조 감소)

val NLL이 지속적으로 감소하여 학습 자체는 성공적으로 수행되었다. 그러나 이상 탐지 성능(AUROC)이 낮은 것은 학습 수렴 문제가 아닌 모델 구조 자체의 한계에 기인한다 (9장 참고).

---

## 8. Ablation 실험

체형 조건화와 모델 구조 각 요소의 기여도를 분리 검증하기 위해 5종의 변형 모델을 설계·학습하였다.

### 8.1 Ablation 변형 설명

**① no_cond — 체형 Conditioning 제거**

BC-STNF에서 FiLM 레이어와 Body Encoder를 제거하고, RealNVP에도 c를 주입하지 않는 버전이다. 즉, 체형 정보 없이 포즈만으로 정상 분포를 학습한다.

- **목적**: 체형 conditioning(FiLM)이 이상 탐지 성능에 기여하는지 확인
- **기대**: 체형 정보가 유용하다면 with_cond(BC-STNF)가 no_cond보다 높은 AUROC를 보여야 한다

**② mlp_feat — ST-GCN 대신 MLP로 특징 추출**

ST-GCN 전체를 제거하고, pose (64, 33, 3)을 flatten한 뒤 MLP로 D=132 특징 벡터를 추출하는 버전이다. 이후 RealNVP 구조는 동일하게 유지한다.

- **목적**: ST-GCN(그래프 구조 활용)이 MLP 대비 이상 탐지에 필요한지 확인
- **기대**: ST-GCN이 효과적이라면 BC-STNF > mlp_feat 순서여야 한다

**③ cluster_cond — 체형을 연속값 대신 클러스터 Discrete Encoding**

체형 7차원 벡터를 K-means(k=8) 클러스터로 분류하고, 각 클러스터를 one-hot embedding으로 조건화하는 버전이다.

- **목적**: 연속적인 체형 벡터 vs 이산적인 체형 카테고리, 어느 것이 효과적인지 비교
- **기대**: 파워리프팅에 체형 유형이 몇 가지로 대별된다면 cluster_cond가 유리할 수 있다

**④ per_exercise — 종목별 분리 모델 vs 통합 모델**

4종목을 하나의 통합 모델이 아닌 종목별로 개별 모델(bc_stnf_squat, bc_stnf_bench, bc_stnf_deadlift, bc_stnf_ohp)을 각각 학습하는 버전이다.

- **목적**: 종목별 특화 모델과 통합 모델의 성능 비교
- **기대**: 종목 간 자세 패턴 차이가 크다면 per_exercise 모델이 유리할 수 있다

**⑤ no_body_cvae — 체형 없는 CVAE**

CVAE에서 체형 벡터 입력을 제거한 버전. Encoder 입력이 `[pose_flat (6336)]`만 되고, Decoder도 z만 받는다.

- **목적**: BC-STNF 계열과 별개로, CVAE에서의 체형 조건화 효과를 분리 검증
- **기대**: 체형 조건화가 유효하다면 CVAE(체형有) val loss < NoBodyCVAE(체형無) val loss

### 8.2 Ablation 결과 및 해석

결과는 9장 평가 결과에서 상세히 분석한다.

---

## 9. 평가 결과

### 9.1 평가 지표

**AUROC (Area Under the ROC Curve)**

ROC 곡선(False Positive Rate vs True Positive Rate)의 아래 면적이다. threshold에 무관하게 모델의 전반적인 정상/이상 분류 능력을 측정한다. 1.0이 완벽한 분류, 0.5는 랜덤 수준이다. 불균형 데이터에서 threshold 설정과 무관하게 비교할 수 있어 이상 탐지 평가의 기본 지표로 사용된다.

**PR-AUC (Area Under the Precision-Recall Curve)**

Precision-Recall 곡선의 아래 면적이다. 정상 샘플보다 이상 샘플이 훨씬 적은 불균형 데이터에서 AUROC보다 더 민감한 지표다. 이상 탐지처럼 이상 클래스가 중요한 경우 AUROC와 함께 사용한다.

**EER (Equal Error Rate)**

FAR(False Accept Rate) = FRR(False Reject Rate)인 지점의 오류율이다. 낮을수록 좋으며, 임계값 설정이 어려운 상황에서 모델의 균형 성능을 나타내는 지표다.

### 9.2 실제 Tier2 평가 결과

| 모델 | AUROC | PR-AUC | EER |
|------|-------|--------|-----|
| BC-STNF (with_cond) | 0.5112 | 0.4986 | 0.4746 |
| no_cond | 0.5866 | 0.5524 | 0.4576 |
| CVAE | 0.4800 | 0.4879 | 0.4576 |
| mlp_feat | 0.5513 | 0.5670 | 0.4746 |
| cluster_cond | 0.5967 | 0.6028 | 0.4492 |

모든 모델에서 AUROC ≈ 0.5로 랜덤 수준에 수렴하였다.

**원인 분석**: 학습 데이터(Tier 1)와 평가 데이터(Tier 2) 모두 YouTube에서 수집하였다. 유튜브에 운동 영상을 올리는 일반인들도 어느 정도의 자세를 갖추고 있어, Tier 1 선수 영상과 Tier 2 일반인 이상 영상 간의 분포 차이가 충분하지 않았다. STG-NF가 보행 이상 탐지에서 성공한 것은 보행 이상(쓰러짐, 비정상 행동)이 정상 보행과 분포가 명확히 분리되기 때문이다. 파워리프팅의 자세 오류는 그 경계가 훨씬 미묘하다.

이 한계를 확인한 후, 학습 분포 기반의 Synthetic OOD 평가를 추가로 수행하였다.

### 9.3 Synthetic OOD 평가

**Synthetic OOD 생성 방법**: 학습 데이터의 분포를 추정하여, 그 분포에서 k=8.0σ 이상 벗어난 포즈를 이상(OOD) 샘플로 생성한다. 이를 통해 모델이 "정상 분포에서 명확히 벗어난 포즈"를 탐지할 수 있는지 통제된 환경에서 측정한다.

```python
mean, std = train_distribution.statistics()
abnormal_pose = normal_pose + k × std × noise  # k=8.0
```

정상 200 : 이상 200으로 balanced 평가를 수행하였다.

| 모델 | AUROC | 비고 |
|------|-------|------|
| CVAE (체형 있음) | **0.9528** | 최고 성능 |
| CVAE (체형 없음) | 0.9532 | ≈ 동일 |
| cluster_cond | **0.9106** | BC-STNF 계열 최고 |
| mlp_feat | 0.6967 | ST-GCN 제거 효과 |
| no_cond | 0.5866 | |
| BC-STNF (with_cond) | 0.5002 | 이상 탐지 실패 |

### 9.4 결과 분석

**BC-STNF 실패 원인**

1. **ST-GCN 공간 스무딩**: ST-GCN은 원래 액션 인식(행동 분류)을 위해 설계된 구조로, 인접 관절의 신호를 집계하여 공통 패턴을 추출하는 것이 목표다. 이 과정에서 특정 관절의 국소 변형 신호(예: 무릎 cave-in)가 인접 관절 신호와 평균화되어 희석된다. 이상 탐지는 반대로 이러한 국소 신호를 강조해야 하므로 ST-GCN이 근본적으로 부적합하였다.

2. **Flow coupling collapse**: L2 regularization(λ=1e-2)이 scale 파라미터 s를 0 방향으로 압박하여, log_det가 상수에 가까워진다. 이렇게 되면 Flow가 입력과 무관한 near-identity 변환으로 퇴화하여 이상 탐지 능력을 잃는다.

**Ablation 인사이트**

- `mlp_feat`(ST-GCN 제거, MLP로 대체, AUROC 0.6967)이 `BC-STNF`(0.5002)보다 높다 → ST-GCN 자체가 이상 탐지의 병목
- `cluster_cond`(0.9106)이 `BC-STNF`(0.5002)보다 높다 → Flow 구조보다 체형 인코딩 방식의 영향이 큼
- `CVAE`(0.9528)이 `mlp_feat`(0.6967)보다 높다 → exact NLL(Flow) 방식보다 재구성 오류(VAE) 방식이 이상 탐지에 더 robust

**체형 조건화 효과**

Synthetic OOD에서 CVAE 체형 있음(0.9528)과 없음(0.9532)의 AUROC 차이가 미미하다. 이는 Synthetic OOD가 포즈 좌표만 변형하고 체형 벡터는 고정하기 때문에, 체형 conditioning의 변별력이 발휘되지 않는 구조적 한계가 있다.

그러나 **val loss 비교에서 체형 있음 0.7478 vs 체형 없음 0.9587**로 명확한 차이가 있다. 이는 체형 정보가 포즈의 정상 분포를 더 타이트하게 모델링하는 데 기여함을 보여준다. 실제 다양한 체형의 사용자가 동일한 운동을 수행하는 데이터셋에서는 체형 조건화의 효과가 더 뚜렷하게 나타날 것으로 기대한다.

**핵심 결론**: 체형 조건화 아이디어 자체는 유효하다(val loss로 검증). 다만 BC-STNF(ST-GCN + Flow)보다 CVAE(MLP + 재구성 오류)가 파워리프팅 이상 탐지에 더 적합한 구조임을 실험을 통해 발견하였다.

---

## 10. 추론 파이프라인 및 배포

### 10.1 End-to-End 추론 파이프라인

```
영상 업로드 (mp4, 720p+)
    ↓
① MediaPipe BlazePose  →  (T, 33, 4)
    ↓
② 체형 벡터 추출 (정적 구간 keypoint → body (7,))
   + 이상치 클리핑 (abs > 100 → clip to [-10, 10])
   + 학습 시 저장된 body_mean / body_std 로 정규화
    ↓
③ rep 분할 + 64프레임 resample
   segment_reps(kps, exercise)  →  N개 rep (각 64, 33, 3)
    ↓
④ CVAE anomaly_score(pose, body)  →  이상 점수 (scalar per rep)
   CVAE joint_attribution(pose, body)  →  heatmap (64, 33)
    ↓
⑤ heatmap → 시간축 mean → (33,) → 상위 3개 관절 추출
    ↓
⑥ GPT-4o API  →  종목·이상 점수·문제 관절 기반 한국어 피드백 2~3문장
    ↓
⑦ FastAPI JSON 응답  →  HTML 프론트엔드 표시
```

체형 벡터 body(7,)는 추론 시에도 반드시 필요하다. 영상에서 체형 벡터를 자동 추출하고, 학습 시 계산된 body_mean / body_std로 정규화하여 모델에 입력한다. 학습과 추론이 동일한 정규화 기준을 사용해야 한다.

### 10.2 이상 판정 기준 (Threshold)

임의의 threshold를 설정하는 대신, val set의 95th percentile을 threshold로 사용한다.

```python
# scripts/calc_threshold.py
val_scores = [model.anomaly_score(pose, body) for pose, body in val_set]
threshold_95 = np.percentile(val_scores, 95)  # → 1.8715
```

val set 9,052 rep에 대해 계산한 threshold = **1.8715**. 이 값은 checkpoint에 저장되어 추론 시 자동으로 로드된다. 점수가 threshold를 초과하면 이상(Anomaly)으로 판정한다. 95th percentile을 사용함으로써 학습 분포 내 5%의 어려운 정상 케이스를 제외한 기준을 설정한다.

### 10.3 FastAPI 서버 및 웹 프론트엔드

FastAPI 서버(`app.py`)가 `/analyze` POST 엔드포인트를 제공하며, 멀티파트 폼 데이터로 영상 파일과 종목 정보를 받는다. 종목별 체크포인트를 lazy loading하여 메모리를 절약한다.

HTML 프론트엔드(`static/index.html`)는 드래그앤드롭 영상 업로드, 분석 진행 표시, rep별 결과(이상 점수·문제 관절·자연어 피드백) 시각화를 제공한다.

### 10.4 Docker 배포

```dockerfile
FROM python:3.11-slim
# torch CPU wheel 명시적 설치 (컨테이너 환경 호환)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt
RUN pip install fastapi "uvicorn[standard]" python-multipart python-dotenv
```

```bash
# 실행
OPENAI_API_KEY=sk-... docker-compose up --build
# http://localhost:8000 접속
```

checkpoints/는 볼륨 마운트로 컨테이너 외부에서 관리한다. `demo/` 폴더의 종목별 샘플 영상으로 즉시 테스트할 수 있다.

### 10.5 데모 결과

| 영상 | 종목 | 정상 비율 | 이상 비율 | 비고 |
|------|------|---------|---------|------|
| normal_squat.mp4 | Squat | 60% | 40% | 단일 영상, 혼합 결과 |
| normal_bench.mp4 | Bench | 80% | 20% | 단일 영상, 혼합 결과 |
| abnormal_deadlift.mp4 | Deadlift | 5/6 rep 정상 | 1/6 rep 이상 | score 2.39 (threshold 초과) |
| normal_ohp.mp4 | OHP | 6/6 rep 정상 | 0 | 평균 점수 1.09 |
| abnormal_ohp.mp4 | OHP | 0 | 2/2 rep 이상 | 평균 점수 45.79 |

threshold = 1.8715 기준으로 정상/이상이 구분된다. 스쿼트·벤치 영상은 정상 자세 중심이나 일부 rep에서 이상이 탐지되어 실제 운동 영상의 자연스러운 자세 변동을 반영한다.

---

## 11. 트러블슈팅

프로젝트 진행 중 발생한 주요 기술적 문제와 해결 방법을 기록한다.

### 11.1 MediaPipe API 변경 (mp.solutions 제거)

**이슈**: MediaPipe 0.10+ 버전에서 `mp.solutions.pose` 등 기존 solutions API가 제거되어 `AttributeError` 발생.

**원인**: MediaPipe가 0.10 버전에서 내부 아키텍처를 Tasks API 중심으로 재편성하면서 레거시 solutions API를 공식 제거하였다.

**해결**: Tasks API와 `.task` 모델 파일 방식으로 전면 교체하였다. `pose_landmarker_heavy.task` 파일을 다운로드하여 사용하며, `PoseLandmarker` 클래스로 프레임별 추론을 수행한다.

```python
# 변경 전 (구버전)
mp_pose = mp.solutions.pose.Pose(...)

# 변경 후 (Tasks API)
from mediapipe.tasks.python.vision import PoseLandmarker
landmarker = PoseLandmarker.create_from_options(options)
result = landmarker.detect(mp.Image(image_format=ImageFormat.SRGB, data=frame))
```

### 11.2 crawl.py TypeError

**이슈**: `yt-dlp` 실행 시 `TypeError: 'int' object is not subscriptable` 발생.

**원인**: `--get-duration` 플래그와 `--print` 플래그를 동시에 사용하면 yt-dlp가 duration을 별도의 숫자 단독 라인으로 출력한다. 이를 `json.loads()`로 파싱하면 Python int가 반환되어 `item["id"]`로 딕셔너리 접근 시 에러가 발생한다.

**해결**: `--get-*` 플래그를 전부 제거하고 `--print "%(id)s\t%(title)s\t%(duration)s"` 형식으로 단일 라인 출력으로 통일하였다.

### 11.3 크롤링 재실행 시 max_videos 초과

**이슈**: 크롤링 스크립트를 재실행하면 기존에 다운로드한 파일보다 max_videos만큼 더 다운로드됨.

**원인**: `downloaded = 0`으로 초기화하여 기존 파일을 카운트에 포함하지 않았다.

**해결**: 스크립트 시작 시 기존 다운로드 파일 수를 먼저 세어 초기화하였다.
```python
existing = list(out_dir.glob("*.mp4"))
downloaded = len(existing)  # 기존 파일 수로 초기화
```

### 11.4 BC-STNF OOM (Out of Memory)

**이슈**: BC-STNF 학습 중 메모리 부족으로 프로세스가 강제 종료됨 (스왑 포함 38GB 초과).

**원인**: 초기 설계에서 ST-GCN 출력 `(B, 64, 33, 64)` (채널 64개)을 전체 flatten하면 D=64×33×64=135,168이 된다. 이를 RealNVP의 Linear 레이어에 입력하면 weight 행렬이 135,168×135,168 크기가 되어 18GB+ 메모리가 필요하다.

**해결**: ST-GCN 채널 수를 3→8→4로 줄이고, Temporal Mean Pooling으로 D=132까지 축소하였다.

```python
# 변경 전: D = 64 × 33 × 64 = 135,168
# 변경 후: D = 33 × 4 = 132 (채널 4, 시간축 mean pooling)
feat = feat.mean(dim=1)   # (B, 64, 33, 4) → (B, 33, 4)
feat = feat.reshape(B, -1)  # → (B, 132)
```

### 11.5 BC-STNF .view() 오류

**이슈**: ST-GCN Block 1 통과 후 `.view()` 호출 시 `RuntimeError: view size is not compatible` 발생.

**원인**: STGCNBlock 내부의 permute 연산 후 텐서가 non-contiguous 상태가 되어 `.view()`를 바로 사용할 수 없다.

**해결**: `.view()` 대신 `.reshape()`으로 교체하였다. `.reshape()`은 non-contiguous 텐서에 대해 자동으로 contiguous 복사본을 만들어 처리한다.

### 11.6 체형 벡터 이상치

**이슈**: bench 종목에서 1,140개, deadlift 8개, ohp 11개의 체형 벡터 값이 수백만에 달하는 이상치 발생.

**원인**: 카메라가 정면 또는 머리 위쪽 각도로 촬영된 영상에서 다리 keypoint의 2D 투영 거리가 0에 가까워진다. 이를 분모로 사용하는 비율 계산에서 b[0](thigh/shin)과 b[6](대칭도)가 폭발한다.

```python
# 문제가 된 계산
b[0] = thigh_length / (shin_length + 1e-6)  # shin ≈ 0 → b[0] → ∞
```

특히 bench 영상은 바벨을 위에서 촬영하는 경우가 많아 집중적으로 발생하였다. (특정 12개 영상에서 1,000개 이상 이상치 발생)

**해결 (임시)**: 학습 데이터 필터링 단계에서 `np.any(np.abs(body) > 100)` 조건으로 해당 샘플을 제거하였다. 추론 시에는 `np.clip(body, -10, 10)`으로 이상치를 클리핑한다.

### 11.7 이중 정규화 (추론 점수 범위 불일치)

**이슈**: 추론 시 이상 점수가 val set(0.67)과 전혀 다른 범위(100+ 이상)로 계산됨.

**원인**: `inference.py`에서 `normalize(kps)`로 한 번 정규화한 뒤 `segment_reps()`에 전달했는데, `segment_reps()` 내부에서 다시 정규화를 수행하여 이중 정규화가 발생하였다.

**해결**: `inference.py`에서 `normalize()` 호출을 제거하고 raw keypoints를 직접 `segment_reps()`에 전달하도록 수정하였다.

### 11.8 threshold 미설정으로 모든 rep 정상 판정

**이슈**: 프론트엔드에서 모든 rep이 "정상"으로 표시됨.

**원인**: 초기 checkpoint에 `threshold_95` 키가 없어, fallback으로 `thr = score * 1.2`를 사용하였다. 이 경우 threshold가 항상 현재 점수보다 높아 모든 rep이 정상으로 판정된다.

**해결**: `scripts/calc_threshold.py`를 구현하여 val set 9,052개 rep의 95th percentile을 계산하고 checkpoint에 저장하였다. 계산된 threshold = 1.8715.

### 11.9 Docker에서 torch 미설치

**이슈**: Docker 컨테이너 실행 시 `ModuleNotFoundError: No module named 'torch'` 발생.

**원인**: `requirements.txt`에 torch가 포함되지 않았다. 개발 환경에서는 conda로 별도 설치하였으나, Docker 이미지 빌드 시 누락되었다.

**해결**: `Dockerfile`에 torch CPU wheel을 명시적으로 설치하는 명령을 추가하였다.

```dockerfile
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 11.10 inference.py 튜플 언패킹 오류

**이슈**: 추론 실행 시 `AttributeError: 'tuple' object has no attribute 'shape'` 발생.

**원인**: `segment_reps()`가 `(rep_array, start_frame, end_frame)` 형태의 튜플을 반환하는데, `for i, rep in enumerate(reps):`로 반복하면 rep에 튜플 전체가 할당되어 `resample(rep)` 호출 시 오류가 발생하였다.

**해결**: 튜플 언패킹을 명시하였다.
```python
for i, (rep, _, _) in enumerate(reps):
    pose_64 = resample(rep)
```

---

## 12. 결론 및 한계점

### 12.1 결론

본 프로젝트는 체형 조건부 딥러닝으로 빅4 파워리프팅 자세 이상을 탐지하는 시스템을 구현하였다. 총 90,514 rep의 학습 데이터를 수집·전처리하고, CVAE와 BC-STNF를 구현하여 Ablation 5종으로 검증하였다.

**주요 결과**:

1. **CVAE(체형 조건화)** 가 Synthetic OOD 평가에서 AUROC 0.9528을 달성하였다. val loss 비교(체형有 0.7478 vs 체형無 0.9587)를 통해 체형 정보가 포즈 분포 학습의 품질을 향상시킴을 확인하였다.

2. **BC-STNF** 는 val NLL이 단조 감소하여 학습 자체는 수렴하였으나, ST-GCN의 공간 스무딩과 Flow coupling collapse로 인해 이상 탐지 성능(AUROC 0.50)이 낮았다. 이는 ST-GCN이 이상 탐지보다 액션 인식에 적합한 구조임을 시사한다.

3. **Ablation 결과**는 ST-GCN 제거(mlp_feat, AUROC 0.6967) 및 VAE 기반 접근(CVAE, AUROC 0.9528) 이 BC-STNF(0.5002)보다 우수함을 보여, 파워리프팅 이상 탐지에서는 재구성 오류 기반 접근이 exact NLL 기반 접근보다 더 robust함을 확인하였다.

4. 전체 파이프라인(MediaPipe → CVAE → GPT-4o → FastAPI → 웹 프론트엔드)이 Docker 환경에서 end-to-end로 동작함을 확인하였다.

### 12.2 한계점

**① 실제 평가 데이터의 분포 문제**

Tier 2 평가에서 모든 모델이 AUROC ≈ 0.5에 수렴하였다. 이는 학습과 평가 모두 YouTube 영상에서 수집하여 분포가 유사하기 때문이다. 실제 현장에서 수집한 다양한 수준의 운동 영상 데이터셋 구축이 필요하다.

**② 체형 조건화 효과의 직접 검증 한계**

Synthetic OOD 평가가 포즈만 변형하고 체형은 고정하기 때문에, 체형 조건화의 이상 탐지 기여를 직접 측정하기 어렵다. 체형이 다른 사람들이 동일한 운동을 수행하는 실제 데이터셋에서의 추가 검증이 필요하다.

**③ BC-STNF 구조적 한계**

ST-GCN의 공간 스무딩 문제는 설계 단계에서 예측하기 어려웠다. 이상 탐지를 위한 그래프 신경망에서는 스무딩 대신 국소 이상을 강조하는 구조(예: attention 기반, 잔차 신호 분리)가 필요하다.

**④ 카메라 앵글 의존성**

학습 분포와 다른 촬영 각도(예: 정면 촬영 영상에 측면 학습 데이터)에서는 체형 벡터 이상치 및 이상 점수 폭발이 발생할 수 있다. 카메라 앵글 불변 전처리 방법이 필요하다.

**⑤ OHP 데이터 부족**

OHP는 4종목 중 학습 reps(12,697)가 가장 적고 촬영 각도가 다양하지 않아 분포가 좁게 학습되었다. 이로 인해 학습 분포를 약간 벗어난 영상에도 이상 점수가 크게 올라가는 경향이 있다.

---

## 13. 참고 문헌

1. Hirschorn, O., & Avidan, S. (2023). Normalizing Flows for Human Pose Anomaly Detection. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2023)*.
2. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density Estimation using Real NVP. *International Conference on Learning Representations (ICLR 2017)*.
3. Perez, E., Strub, F., De Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2018)*.
4. Yan, S., Xiong, Y., & Lin, D. (2018). Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2018)*.
5. Anthropometry-Aware Deep Learning for Exercise Assessment. *PMC 2025*.

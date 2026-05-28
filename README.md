# BodyFit

**Body-Aware Form Assessment for Big4 Powerlifts**

체형 조건부 딥러닝 모델을 활용한 개인 맞춤형 자세 이상 탐지 시스템.

## Overview

기존 자세 평가 AI는 모든 사용자에게 동일한 기준을 적용하지만, 실제로는 체형(대퇴/경골 비율, 팔 길이 등)에 따라 올바른 자세 기준이 다르다. BodyFit은 체형 정보를 조건 입력으로 활용해 개인별 맞춤 이상 탐지를 수행한다.

## Core Model — BC-STNF

**Body-Conditioned Spatio-Temporal Normalizing Flow**

```
A(x) = -log P(pose | body)
```

- **Body Encoder**: 체형 7차원 벡터 → 16차원 조건 벡터 c
- **ST-GCN + FiLM**: 관절 그래프 기반 시공간 임베딩, c로 channel-wise 조건화
- **Conditional RealNVP × 6**: 가역 변환으로 exact log-likelihood 계산

## Target Exercises

Squat · Bench Press · Deadlift · Overhead Press (Big4 Powerlifts)

## Model Stack

| 모델 | 역할 |
|------|------|
| MediaPipe BlazePose | Pose 추출 |
| CVAE | Phase 1 baseline / fallback |
| BC-STNF | 핵심 이상 탐지 모델 |
| Claude API | 자연어 피드백 생성 |

## Project Structure

```
bodyfit/
├── src/
│   ├── data/          # 크롤링·전처리
│   ├── models/        # CVAE, BC-STNF
│   ├── training/      # 학습 스크립트
│   └── evaluation/    # 평가·ablation
├── notebooks/         # 실험·시각화
├── data/              # (gitignore — 로컬 전용)
├── checkpoints/       # (gitignore — 로컬 전용)
└── results/           # (gitignore — 로컬 전용)
```

## Setup

```bash
# 1. conda 환경 생성 (Python 3.11 필수 — MediaPipe 호환)
conda create -n bodyfit python=3.11 -y
conda activate bodyfit

# 2. PyTorch 설치 (Mac M3 — MPS 백엔드 자동 포함)
pip install torch torchvision torchaudio

# 3. 나머지 패키지
pip install -r requirements.txt
```

## Troubleshooting

### `TypeError: 'int' object is not subscriptable` — crawl.py

**발생 위치**: `src/data/crawl.py` `_search_video_ids()` → `item["id"]`

**원인**: yt-dlp에서 `--get-id`, `--get-title`, `--get-duration`과 `--print`를 동시에 사용하면 충돌 발생.
`--get-duration`이 duration을 별도 숫자 라인(`"120"`)으로 추가 출력하고, `json.loads("120")`이 int를 반환해서 딕셔너리 접근 시 TypeError.

**해결**: `--get-*` 플래그 전부 제거, `--print`만 사용.

```python
# 수정 전 (오류)
cmd = ["yt-dlp", ..., "--get-id", "--get-title", "--get-duration", "--print", '{"id":...}']

# 수정 후 (정상)
cmd = ["yt-dlp", ..., "--print", '{"id":"%(id)s","title":"%(title)s","duration":%(duration)s}']
```

---

## References

- Hirschorn & Avidan. Normalizing Flows for Human Pose Anomaly Detection. ICCV 2023
- Dinh et al. Density Estimation using Real NVP. ICLR 2017
- Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018
- Yan et al. Spatial Temporal Graph Convolutional Networks. AAAI 2018

---

세종대학교 딥러닝 실습 · 2026 Spring · 24013840 이동현

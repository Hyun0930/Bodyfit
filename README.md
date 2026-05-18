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
pip install -r requirements.txt
```

## References

- Hirschorn & Avidan. Normalizing Flows for Human Pose Anomaly Detection. ICCV 2023
- Dinh et al. Density Estimation using Real NVP. ICLR 2017
- Perez et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI 2018
- Yan et al. Spatial Temporal Graph Convolutional Networks. AAAI 2018

---

세종대학교 딥러닝 실습 · 2026 Spring · 24013840 이동현

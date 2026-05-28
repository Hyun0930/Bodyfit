"""평가 지표: AUROC, PR-AUC, EER, Per-joint IoU."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def threshold_from_val(val_scores: np.ndarray, percentile: float = 5) -> float:
    """val set 정상 점수의 하위 percentile → threshold.

    정상 점수의 percentile 이하는 '너무 정상', 이 값보다 높으면 이상으로 판단.
    기본 5th percentile: 정상의 95%를 포함하는 경계.
    """
    return float(np.percentile(val_scores, 100 - percentile))


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """이상 점수와 라벨로 평가 지표 계산.

    Args:
        scores: (N,) 높을수록 이상 (BC-STNF NLL, CVAE MSE 모두 호환)
        labels: (N,) 0=정상, 1=이상
        threshold: 이 값 초과 시 이상으로 분류

    Returns:
        dict with keys: auroc, pr_auc, eer, accuracy, f1
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    auroc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    preds = (scores > threshold).astype(int)
    accuracy = float((preds == labels).mean())
    f1 = float(f1_score(labels, preds, zero_division=0))

    return {
        "auroc": round(auroc, 4),
        "pr_auc": round(pr_auc, 4),
        "eer": round(eer, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
    }


def per_joint_iou(
    pred_heatmap: np.ndarray,
    gt_heatmap: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """관절별 IoU 계산.

    Args:
        pred_heatmap: (N, 33) 관절별 attribution 점수 (연속값)
        gt_heatmap:   (N, 33) 수동 라벨 (0 or 1)
        threshold:    pred 이진화 기준. None이면 각 샘플 상위 50% 사용.

    Returns:
        dict: {joint_idx(str): iou, "mean_iou": float}
    """
    pred_heatmap = np.asarray(pred_heatmap, dtype=float)
    gt_heatmap = np.asarray(gt_heatmap, dtype=float)

    if threshold is None:
        median = np.median(pred_heatmap, axis=1, keepdims=True)
        pred_bin = (pred_heatmap >= median).astype(int)
    else:
        pred_bin = (pred_heatmap >= threshold).astype(int)

    gt_bin = (gt_heatmap > 0).astype(int)

    result = {}
    ious = []
    for j in range(33):
        p = pred_bin[:, j]
        g = gt_bin[:, j]
        intersection = (p & g).sum()
        union = (p | g).sum()
        iou = float(intersection / union) if union > 0 else 1.0
        result[str(j)] = round(iou, 4)
        ious.append(iou)

    result["mean_iou"] = round(float(np.mean(ious)), 4)
    return result


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: str,
    threshold: float | None = None,
    label_fn=None,
) -> dict:
    """모델 + DataLoader → metrics dict.

    Args:
        model:       anomaly_score(pose, body) → (B,) 를 가진 모델
        data_loader: (pose, body) 또는 (pose, body, label) 반환
        device:      'cpu' | 'cuda' | 'mps'
        threshold:   None이면 val set 95th percentile 사용
        label_fn:    배치에서 label 추출 함수. None이면 모두 0(정상)으로 가정

    Returns:
        dict with auroc, pr_auc, eer, accuracy, f1, threshold
    """
    model.eval()
    model.to(device)

    all_scores, all_labels = [], []

    for batch in data_loader:
        if len(batch) == 3:
            pose, body, label = batch
        else:
            pose, body = batch
            label = torch.zeros(pose.shape[0], dtype=torch.long)

        if label_fn is not None:
            label = label_fn(label)

        pose = pose.to(device)
        body = body.to(device)
        scores = model.anomaly_score(pose, body).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(label.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    if threshold is None:
        normal_scores = all_scores[all_labels == 0]
        threshold = threshold_from_val(normal_scores)

    metrics = compute_metrics(all_scores, all_labels, threshold)
    metrics["threshold"] = round(float(threshold), 4)
    return metrics

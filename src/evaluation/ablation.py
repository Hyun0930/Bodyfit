"""Ablation 5종 실행 및 변형 모델 정의."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_model, threshold_from_val
from src.models.bc_stnf import BCSTNF, FLOW_DIM
from src.models.body_encoder import BodyEncoder
from src.models.cvae import CVAE
from src.models.realnvp import ConditionalRealNVP
from src.models.st_gcn import STGCN

_PRIOR = Normal(0, 1)

EXERCISES = ["squat", "bench", "deadlift", "ohp"]

RAW_FLOW_DIM = 33 * 3  # 99 — 시간 mean pooling 후 flatten


# ---------------------------------------------------------------------------
# Ablation 1: 체형 조건화 제거
# ---------------------------------------------------------------------------

class NoCondBCSTNF(nn.Module):
    """Ablation 1 — body 조건화 없이 동일 구조 유지.

    body 입력을 zeros로 대체하고 FiLM γ=1, β=0으로 고정하는 대신,
    구조를 단순화: body_enc 제거, c를 zeros(16)으로 고정.
    """

    def __init__(self, n_coupling: int = 6):
        super().__init__()
        self.stgcn = STGCN()
        self.flow = ConditionalRealNVP(FLOW_DIM, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        B = pose.shape[0]
        c = torch.zeros(B, 16, device=pose.device)  # 체형 조건화 없음
        feat = self.stgcn(pose, c).mean(dim=1)       # (B, 33, 4) temporal mean pooling
        z, log_det, _ = self.flow(feat.flatten(1), c)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det
        return -log_prob

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(pose, body)


# ---------------------------------------------------------------------------
# Ablation 3: ST-GCN → 단순 MLP
# ---------------------------------------------------------------------------

class MLPFeatureBCSTNF(nn.Module):
    """Ablation 3 — ST-GCN 대신 Linear(6336 → FLOW_DIM) 사용."""

    def __init__(self, n_coupling: int = 6):
        super().__init__()
        self.body_enc = BodyEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(64 * 33 * 3, FLOW_DIM),
            nn.ReLU(),
            nn.Linear(FLOW_DIM, FLOW_DIM),
        )
        self.flow = ConditionalRealNVP(FLOW_DIM, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        c = self.body_enc(body)
        feat = self.mlp(pose.flatten(1))
        z, log_det, _ = self.flow(feat, c)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det
        return -log_prob

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(pose, body)


# ---------------------------------------------------------------------------
# Ablation 4: Continuous → Cluster body encoding
# ---------------------------------------------------------------------------

class ClusterBodyEncoder(nn.Module):
    """Ablation 4 — k-means 클러스터 할당 one-hot → 16차원 임베딩."""

    def __init__(self, n_clusters: int = 10):
        super().__init__()
        self.n_clusters = n_clusters
        self.register_buffer("centroids", torch.zeros(n_clusters, 7))
        self.embed = nn.Linear(n_clusters, 16, bias=False)
        self._fitted = False

    def fit(self, body_data: np.ndarray):
        """학습 데이터의 body 벡터로 k-means 적합."""
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        km.fit(body_data)
        self.centroids.copy_(torch.tensor(km.cluster_centers_, dtype=torch.float32))
        self._fitted = True

    def forward(self, body: torch.Tensor) -> torch.Tensor:
        # 유클리드 거리 기반 소속 클러스터 → one-hot → embed
        dists = torch.cdist(body, self.centroids)  # (B, K)
        one_hot = torch.zeros_like(dists).scatter_(1, dists.argmin(dim=1, keepdim=True), 1.0)
        return self.embed(one_hot)


class ClusterCondBCSTNF(nn.Module):
    """Ablation 4 — ClusterBodyEncoder 사용 BC-STNF."""

    def __init__(self, n_clusters: int = 10, n_coupling: int = 6):
        super().__init__()
        self.body_enc = ClusterBodyEncoder(n_clusters)
        self.stgcn = STGCN()
        self.flow = ConditionalRealNVP(FLOW_DIM, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        c = self.body_enc(body)
        feat = self.stgcn(pose, c).mean(dim=1)       # (B, 33, 4) temporal mean pooling
        z, log_det, _ = self.flow(feat.flatten(1), c)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det
        return -log_prob

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(pose, body)


# ---------------------------------------------------------------------------
# Raw Pose Flow — ST-GCN 제거 + s_reg=0
# ---------------------------------------------------------------------------

class RawPoseFlow(nn.Module):
    """ST-GCN 없이 raw pose → time mean → flatten → conditional RealNVP.

    두 문제 동시 해결:
      1) ST-GCN 공간 스무딩 제거 → 관절 변형 신호 직접 전달
      2) s_reg_lambda=0 → coupling s가 0으로 수렴하는 collapse 방지
    """

    def __init__(self, n_coupling: int = 6):
        super().__init__()
        self.body_enc = BodyEncoder()                                          # 7 → 16
        self.proj = nn.Linear(RAW_FLOW_DIM, 100)                              # 99 → 100 (짝수)
        self.flow = ConditionalRealNVP(100, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        c = self.body_enc(body)
        feat = self.proj(pose.mean(dim=1).flatten(1))  # (B,64,33,3) → (B,99) → (B,100)
        z, log_det, _ = self.flow(feat, c)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det
        return -log_prob                       # s_reg 없음

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(pose, body)


# ---------------------------------------------------------------------------
# Ablation 실행기
# ---------------------------------------------------------------------------

def _load_model(
    model: nn.Module, ckpt_path: Path, device: str
) -> tuple[bool, torch.Tensor | None, torch.Tensor | None]:
    """checkpoint 로드 → (성공여부, body_mean, body_std) 반환."""
    if not ckpt_path.exists():
        return False, None, None
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    body_mean = ckpt.get("body_mean")
    body_std = ckpt.get("body_std")
    if body_mean is not None:
        body_mean = body_mean.to(device)
    if body_std is not None:
        body_std = body_std.to(device)
    return True, body_mean, body_std


def _collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    body_mean: torch.Tensor | None = None,
    body_std: torch.Tensor | None = None,
) -> np.ndarray:
    model.eval()
    model.to(device)
    scores = []
    with torch.no_grad():
        for batch in loader:
            pose, body = batch[0].to(device), batch[1].to(device)
            if body_mean is not None and body_std is not None:
                body = (body - body_mean) / body_std
            scores.append(model.anomaly_score(pose, body).cpu().numpy())
    return np.concatenate(scores)


def run_ablation(
    ckpt_dir: Path,
    data_root: Path,
    tier3_root: Path | None = None,
    labels_path: Path | None = None,
    device: str = "cpu",
    exercises: list[str] | None = None,
    batch_size: int = 32,
    max_per_class: int | None = None,
) -> dict:
    """5종 ablation 실행 → 결과 dict 반환 + results/ablation.json 저장.

    threshold는 BodyFitDataset val set(정상만)으로 계산.
    AUROC/PR-AUC/EER는 Tier3Dataset(정상+이상)으로 계산.
    checkpoint가 없는 항목은 'no_checkpoint' 로 표시.
    """
    from src.data import BodyFitDataset
    from src.data.tier3_dataset import Tier3Dataset

    exercises = exercises or EXERCISES
    ckpt_dir = Path(ckpt_dir)
    results = {}

    # threshold 계산용: BodyFitDataset val set (정상만)
    train_ds = BodyFitDataset(exercises=exercises, root=data_root)
    _, val_ds = train_ds.split(train_ratio=0.9)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 평가용: Tier3Dataset (정상+이상 라벨 포함)
    t3_root = Path(tier3_root) if tier3_root else Path(data_root).parent / "test"
    t3_labels = Path(labels_path) if labels_path else t3_root / "labels.json"
    tier3_ds = Tier3Dataset(root=t3_root, labels_path=t3_labels, max_per_class=max_per_class)
    tier3_loader = DataLoader(tier3_ds, batch_size=batch_size, shuffle=False)
    print(f"Tier3 평가셋: {tier3_ds.label_counts()}")

    # ---- Ablation 1: 체형 조건화 유무 ----
    print("[Ablation 1] 체형 조건화 유무")
    bc_model = BCSTNF()
    no_cond = NoCondBCSTNF()
    bc_ok, bc_mean, bc_std = _load_model(bc_model, ckpt_dir / "bc_stnf" / "best.pt", device)
    nc_ok, nc_mean, nc_std = _load_model(no_cond, ckpt_dir / "no_cond" / "best.pt", device)
    if bc_ok and nc_ok:
        thr = threshold_from_val(_collect_scores(bc_model, val_loader, device, bc_mean, bc_std))
        results["ablation1_with_cond"] = evaluate_model(bc_model, tier3_loader, device, thr, body_mean=bc_mean, body_std=bc_std)
        nc_thr = threshold_from_val(_collect_scores(no_cond, val_loader, device, nc_mean, nc_std))
        results["ablation1_no_cond"] = evaluate_model(no_cond, tier3_loader, device, nc_thr, body_mean=nc_mean, body_std=nc_std)
    else:
        results["ablation1"] = "no_checkpoint"
        print("  → checkpoint 없음, skip")

    # ---- Ablation 2: CVAE vs BC-STNF ----
    print("[Ablation 2] CVAE vs BC-STNF")
    cvae = CVAE()
    cvae_ok, cvae_mean, cvae_std = _load_model(cvae, ckpt_dir / "cvae" / "best.pt", device)
    if bc_ok and cvae_ok:
        thr = threshold_from_val(_collect_scores(bc_model, val_loader, device, bc_mean, bc_std))
        results["ablation2_bc_stnf"] = evaluate_model(bc_model, tier3_loader, device, thr, body_mean=bc_mean, body_std=bc_std)
        cvae_thr = threshold_from_val(_collect_scores(cvae, val_loader, device, cvae_mean, cvae_std))
        results["ablation2_cvae"] = evaluate_model(cvae, tier3_loader, device, cvae_thr, body_mean=cvae_mean, body_std=cvae_std)
    else:
        results["ablation2"] = "no_checkpoint"
        print("  → checkpoint 없음, skip")

    # ---- Ablation 3: ST-GCN vs MLP ----
    print("[Ablation 3] ST-GCN vs MLP")
    mlp_model = MLPFeatureBCSTNF()
    mlp_ok, mlp_mean, mlp_std = _load_model(mlp_model, ckpt_dir / "mlp_feat" / "best.pt", device)
    if bc_ok and mlp_ok:
        thr = threshold_from_val(_collect_scores(bc_model, val_loader, device, bc_mean, bc_std))
        results["ablation3_stgcn"] = evaluate_model(bc_model, tier3_loader, device, thr, body_mean=bc_mean, body_std=bc_std)
        mlp_thr = threshold_from_val(_collect_scores(mlp_model, val_loader, device, mlp_mean, mlp_std))
        results["ablation3_mlp"] = evaluate_model(mlp_model, tier3_loader, device, mlp_thr, body_mean=mlp_mean, body_std=mlp_std)
    else:
        results["ablation3"] = "no_checkpoint"
        print("  → checkpoint 없음, skip")

    # ---- Ablation 4: Cluster vs Continuous body encoding ----
    print("[Ablation 4] Cluster vs Continuous body encoding")
    cluster_model = ClusterCondBCSTNF()
    cl_ok, cl_mean, cl_std = _load_model(cluster_model, ckpt_dir / "cluster_cond" / "best.pt", device)
    if bc_ok and cl_ok:
        thr = threshold_from_val(_collect_scores(bc_model, val_loader, device, bc_mean, bc_std))
        results["ablation4_continuous"] = evaluate_model(bc_model, tier3_loader, device, thr, body_mean=bc_mean, body_std=bc_std)
        cl_thr = threshold_from_val(_collect_scores(cluster_model, val_loader, device, cl_mean, cl_std))
        results["ablation4_cluster"] = evaluate_model(cluster_model, tier3_loader, device, cl_thr, body_mean=cl_mean, body_std=cl_std)
    else:
        results["ablation4"] = "no_checkpoint"
        print("  → checkpoint 없음, skip")

    # ---- Ablation 5: 종목별 분리 vs 통합 모델 ----
    print("[Ablation 5] 종목별 분리 vs 통합 모델")
    if bc_ok:
        thr = threshold_from_val(_collect_scores(bc_model, val_loader, device, bc_mean, bc_std))
        results["ablation5_unified"] = evaluate_model(bc_model, tier3_loader, device, thr, body_mean=bc_mean, body_std=bc_std)
        per_ex_metrics = {}
        for ex in exercises:
            ex_train = BodyFitDataset(exercises=[ex], root=data_root)
            _, ex_val = ex_train.split(train_ratio=0.9)
            ex_val_loader = DataLoader(ex_val, batch_size=batch_size, shuffle=False)
            ex_tier3 = Tier3Dataset(root=t3_root, labels_path=t3_labels, exercises=[ex])
            ex_tier3_loader = DataLoader(ex_tier3, batch_size=batch_size, shuffle=False)
            ex_model = BCSTNF()
            ex_ckpt = ckpt_dir / f"bc_stnf_{ex}" / "best.pt"
            ex_ok, ex_mean, ex_std = _load_model(ex_model, ex_ckpt, device)
            if ex_ok:
                ex_thr = threshold_from_val(_collect_scores(ex_model, ex_val_loader, device, ex_mean, ex_std))
                per_ex_metrics[ex] = evaluate_model(ex_model, ex_tier3_loader, device, ex_thr, body_mean=ex_mean, body_std=ex_std)
            else:
                per_ex_metrics[ex] = "no_checkpoint"
        results["ablation5_per_exercise"] = per_ex_metrics
    else:
        results["ablation5"] = "no_checkpoint"
        print("  → checkpoint 없음, skip")

    # 결과 저장
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {out_path}")

    return results


def _print_table(results: dict):
    print("\n=== Ablation 결과 ===")
    for key, val in results.items():
        if isinstance(val, dict) and "auroc" in val:
            print(f"  {key:35s} AUROC={val['auroc']:.4f}  PR-AUC={val['pr_auc']:.4f}  EER={val['eer']:.4f}")
        elif isinstance(val, str):
            print(f"  {key:35s} {val}")
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict) and "auroc" in sub_val:
                    print(f"  {key}/{sub_key:25s} AUROC={sub_val['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--tier3_root", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exercises", nargs="+", default=EXERCISES)
    parser.add_argument("--max_per_class", type=int, default=None)
    args = parser.parse_args()

    res = run_ablation(
        args.ckpt_dir, args.data_root,
        tier3_root=args.tier3_root,
        labels_path=args.labels_path,
        device=args.device,
        exercises=args.exercises,
        max_per_class=args.max_per_class,
    )
    _print_table(res)

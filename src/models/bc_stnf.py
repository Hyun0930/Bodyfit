import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.body_encoder import BodyEncoder
from src.models.st_gcn import STGCN
from src.models.realnvp import ConditionalRealNVP

_PRIOR = Normal(0, 1)
# ST-GCN 출력 (B,64,33,4) → 시간축 mean pooling → (B,33,4) → flatten
FLOW_DIM = 33 * 4  # 132


class BCSTNF(nn.Module):
    """Body-Conditioned Spatio-Temporal Normalizing Flow

    A(x) = -log P(pose | body)
    ST-GCN 출력을 시간축 mean pooling 후 flow 적용 (D=132)
    """

    def __init__(self, n_coupling: int = 6):
        super().__init__()
        self.body_enc = BodyEncoder()                              # 7 → 16
        self.stgcn = STGCN()                                      # (B,64,33,3) → (B,64,33,4)
        self.flow = ConditionalRealNVP(FLOW_DIM, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor, s_reg_lambda: float = 1e-2) -> torch.Tensor:
        """NLL + L2 reg(s) 계산 — 학습 손실.

        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
            s_reg_lambda: coupling s² 에 대한 L2 계수 (mode collapse 방지, 계획서 명시)
        Returns:
            loss: (B,)  — NLL + λ·Σs²
        """
        c = self.body_enc(body)                                          # (B, 16)
        feat = self.stgcn(pose, c)                                       # (B, 64, 33, 4)
        feat = feat.mean(dim=1)                                          # (B, 33, 4) 시간축 mean pooling
        z, log_det, s_sqsum = self.flow(feat.flatten(1), c)              # z:(B,132), log_det:(B,), s_sqsum:(B,)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det              # (B,)
        nll = -log_prob
        return nll + s_reg_lambda * s_sqsum                              # L2 reg: s→0 방향 압력

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """추론 시 순수 NLL만 반환 (reg 없음)."""
        with torch.no_grad():
            c = self.body_enc(body)
            feat = self.stgcn(pose, c).mean(dim=1)
            z, log_det, _ = self.flow(feat.flatten(1), c)
            log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det
            return -log_prob

    def joint_attribution(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """gradient norm w.r.t. pose → heatmap.

        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
        Returns:
            heatmap: (B, 64, 33)  — 값이 클수록 해당 관절/프레임 기여도 높음
        """
        pose = pose.detach().requires_grad_(True)
        score = self.forward(pose, body).sum()
        score.backward()
        return pose.grad.norm(dim=-1)                         # (B, 64, 33)

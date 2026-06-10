import torch
import torch.nn as nn
from torch.distributions import Normal

from src.models.body_encoder import BodyEncoder
from src.models.st_gcn import STGCN
from src.models.realnvp import ConditionalRealNVP

_PRIOR = Normal(0, 1)
# ST-GCN 출력 (B,64,33,4) flatten → Flow 입력 차원
FLOW_DIM = 64 * 33 * 4  # 8,448


class BCSTNF(nn.Module):
    """Body-Conditioned Spatio-Temporal Normalizing Flow

    A(x) = -log P(pose | body)
    """

    def __init__(self, n_coupling: int = 6):
        super().__init__()
        self.body_enc = BodyEncoder()                              # 7 → 16
        self.stgcn = STGCN()                                      # (B,64,33,3) → (B,64,33,4)
        self.flow = ConditionalRealNVP(FLOW_DIM, c_dim=16, n_coupling=n_coupling)

    def forward(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """NLL 계산 — 학습 손실 및 이상 점수.

        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
        Returns:
            anomaly_score: (B,)  — -log P(pose|body)
        """
        c = self.body_enc(body)                                   # (B, 16)
        feat = self.stgcn(pose, c)                                # (B, 64, 33, 4)
        x = feat.flatten(1)                                       # (B, 8448)
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        z, log_det = self.flow(x, c)                              # z: (B,8448), log_det: (B,)
        log_prob = _PRIOR.log_prob(z).sum(dim=-1) + log_det       # (B,)
        return -log_prob

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(pose, body)

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

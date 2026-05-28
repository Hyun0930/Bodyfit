import torch
import torch.nn as nn


class BodyEncoder(nn.Module):
    """체형 7차원 벡터 → 16차원 조건 벡터 c"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.LayerNorm(16),
        )

    def forward(self, body: torch.Tensor) -> torch.Tensor:
        """
        Args:
            body: (B, 7)
        Returns:
            c: (B, 16)
        """
        return self.net(body)

import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe BlazePose 33 keypoint 해부학적 연결
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),          # 얼굴 우측
    (0, 4), (4, 5), (5, 6), (6, 8),           # 얼굴 좌측
    (9, 10),                                    # 입
    (11, 12),                                   # 어깨
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # 왼팔
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # 오른팔
    (11, 23), (12, 24), (23, 24),              # 몸통
    (23, 25), (25, 27), (27, 29), (27, 31),   # 왼다리
    (24, 26), (26, 28), (28, 30), (28, 32),   # 오른다리
]
N_JOINTS = 33


def _build_adj(n: int, edges: list) -> torch.Tensor:
    """대칭 정규화 인접행렬 (self-loop 포함)."""
    A = torch.zeros(n, n)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    A += torch.eye(n)
    deg = A.sum(dim=1, keepdim=True).clamp(min=1).sqrt()
    return A / deg / deg.T


class FiLM(nn.Module):
    """channel-wise scale·shift: feat = feat · γ(c) + β(c)"""

    def __init__(self, c_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(c_dim, feat_dim * 2)

    def forward(self, feat: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, T, J, C)
            c:    (B, c_dim)
        """
        gamma, beta = self.proj(c).chunk(2, dim=-1)          # (B, C)
        gamma = gamma[:, None, None, :]                       # (B,1,1,C)
        beta  = beta[:, None, None, :]
        return feat * gamma + beta


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, c_dim: int = 16):
        super().__init__()
        self.register_buffer("A", _build_adj(N_JOINTS, EDGES))  # (33,33)

        # GCN weight: (C_in, C_out)
        self.gcn_w = nn.Linear(c_in, c_out, bias=False)

        # TCN: 시간 축 Conv (J를 batch dim으로 처리)
        self.tcn = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=9, padding=4, groups=1),
            nn.BatchNorm1d(c_out),
        )

        self.film = FiLM(c_dim, c_out)
        self.relu = nn.ReLU()

        self.shortcut = nn.Linear(c_in, c_out, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, J, C_in)
            c: (B, 16)
        Returns:
            (B, T, J, C_out)
        """
        B, T, J, C_in = x.shape
        residual = self.shortcut(x)

        # GCN: A × x × W  (per time step)
        h = x.reshape(B * T, J, C_in)        # (B*T, J, C_in)
        h = torch.bmm(self.A.unsqueeze(0).expand(B * T, -1, -1), h)  # (B*T, J, C_in)
        h = self.gcn_w(h)                     # (B*T, J, C_out)
        h = h.view(B, T, J, -1)              # (B, T, J, C_out)
        C_out = h.shape[-1]

        # TCN: Conv1d on T axis, treating B*J as batch
        h = h.permute(0, 2, 3, 1).reshape(B * J, C_out, T)  # (B*J, C_out, T)
        h = self.tcn(h)                                        # (B*J, C_out, T)
        h = h.reshape(B, J, C_out, T).permute(0, 3, 1, 2)    # (B, T, J, C_out)

        h = self.relu(h)
        h = self.film(h, c)
        return self.relu(h + residual)


class STGCN(nn.Module):
    """ST-GCN × 2: pose(B,64,33,3) → feat(B,64,33,4)"""

    def __init__(self, c_dim: int = 16):
        super().__init__()
        self.block1 = STGCNBlock(3, 8, c_dim)
        self.block2 = STGCNBlock(8, 4, c_dim)

    def forward(self, pose: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, 64, 33, 3)
            c:    (B, 16)
        Returns:
            feat: (B, 64, 33, 4)
        """
        x = self.block1(pose, c)
        x = self.block2(x, c)
        return x

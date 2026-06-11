import torch
import torch.nn as nn


class ActNorm(nn.Module):
    """Channel-wise learnable affine (identity 초기화, gradient로 정규화 학습).

    data-dependent init 없이 bias=0, log_scale=0에서 출발 → NaN 위험 제거.
    학습 중 gradient가 자연스럽게 피처 스케일을 정규화 방향으로 조정.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.log_scale = nn.Parameter(torch.zeros(dim))  # exp(0)=1 → identity 출발

    def forward(self, x: torch.Tensor, c: torch.Tensor = None):
        y = (x + self.bias) * self.log_scale.exp()
        log_det = self.log_scale.sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        return y * (-self.log_scale).exp() - self.bias


class CouplingLayer(nn.Module):
    """Conditional Affine Coupling Layer (zero-init → identity 출발)."""

    def __init__(self, dim: int, c_dim: int = 16, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half + c_dim, half),
            nn.ReLU(),
            nn.Linear(half, half * 2),  # → s, t
        )
        # 마지막 레이어 0 초기화 → 초기 s=0, t=0 (identity mapping)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, D)
            c: (B, c_dim)
        Returns:
            y: (B, D)
            log_det: (B,)
        """
        half = x.shape[1] // 2
        if not self.reverse:
            x_a, x_b = x[:, :half], x[:, half:]
        else:
            x_b, x_a = x[:, :half], x[:, half:]

        st = self.net(torch.cat([x_a, c], dim=-1))
        s, t = st[:, :half], st[:, half:]
        s = torch.tanh(s) * 2.0
        self._last_s_sqsum = s.pow(2).sum(dim=-1)  # L2 reg 계산용

        x_b_out = x_b * s.exp() + t
        log_det = s.sum(dim=-1)

        if not self.reverse:
            y = torch.cat([x_a, x_b_out], dim=-1)
        else:
            y = torch.cat([x_b_out, x_a], dim=-1)

        return y, log_det

    def inverse(self, y: torch.Tensor, c: torch.Tensor):
        half = y.shape[1] // 2
        if not self.reverse:
            y_a, y_b = y[:, :half], y[:, half:]
        else:
            y_b, y_a = y[:, :half], y[:, half:]

        st = self.net(torch.cat([y_a, c], dim=-1))
        s, t = st[:, :half], st[:, half:]
        s = torch.tanh(s) * 2.0

        x_b = (y_b - t) * (-s).exp()

        if not self.reverse:
            return torch.cat([y_a, x_b], dim=-1)
        else:
            return torch.cat([x_b, y_a], dim=-1)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP: ActNorm + Affine Coupling (Glow-style)."""

    def __init__(self, dim: int, c_dim: int = 16, n_coupling: int = 6):
        super().__init__()
        layers = []
        for i in range(n_coupling):
            layers.append(ActNorm(dim))
            layers.append(CouplingLayer(dim, c_dim, reverse=(i % 2 == 1)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, D)
            c: (B, c_dim)
        Returns:
            z: (B, D)
            log_det: (B,)
            s_sqsum: (B,)  — coupling layer s² 합계 (L2 reg용)
        """
        log_det = torch.zeros(x.shape[0], device=x.device)
        s_sqsum = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z, c)
            log_det += ld
            if isinstance(layer, CouplingLayer):
                s_sqsum += layer._last_s_sqsum
        return z, log_det, s_sqsum

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, c)
        return x

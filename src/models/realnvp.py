import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """Conditional Additive Coupling Layer (log_det=0 → trivial solution 원천 차단)."""

    def __init__(self, dim: int, c_dim: int = 16, reverse: bool = False):
        super().__init__()
        self.reverse = reverse
        half = dim // 2
        self.net = nn.Sequential(
            nn.Linear(half + c_dim, half * 2),
            nn.ReLU(),
            nn.Linear(half * 2, half),  # → t only
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, D)
            c: (B, c_dim)
        Returns:
            y: (B, D)
            log_det: (B,)  — always 0 for additive coupling
        """
        half = x.shape[1] // 2
        if not self.reverse:
            x_a, x_b = x[:, :half], x[:, half:]
        else:
            x_b, x_a = x[:, :half], x[:, half:]

        t = self.net(torch.cat([x_a, c], dim=-1))
        x_b_out = x_b + t
        log_det = torch.zeros(x.shape[0], device=x.device)

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

        t = self.net(torch.cat([y_a, c], dim=-1))
        x_b = y_b - t

        if not self.reverse:
            return torch.cat([y_a, x_b], dim=-1)
        else:
            return torch.cat([x_b, y_a], dim=-1)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP: n_coupling개 coupling layer."""

    def __init__(self, dim: int, c_dim: int = 16, n_coupling: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(dim, c_dim, reverse=(i % 2 == 1))
            for i in range(n_coupling)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x: (B, D)
            c: (B, c_dim)
        Returns:
            z: (B, D)
            log_det: (B,)   — sum of all layer log-determinants
        """
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z, c)
            log_det += ld
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, c)
        return x

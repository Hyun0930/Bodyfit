import torch
import torch.nn as nn
import torch.nn.functional as F

POSE_DIM = 64 * 33 * 3  # 6336


class CVAE(nn.Module):
    def __init__(self, latent_dim: int = 128, beta: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = nn.Sequential(
            nn.Linear(POSE_DIM + 7, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 7, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, POSE_DIM),
        )

    def encode(self, pose_flat: torch.Tensor, body: torch.Tensor):
        h = self.encoder(torch.cat([pose_flat, body], dim=-1))
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, body], dim=-1))

    def forward(self, pose: torch.Tensor, body: torch.Tensor):
        """
        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
        Returns:
            pose_recon: (B, 64, 33, 3)
            mu:         (B, latent_dim)
            log_var:    (B, latent_dim)
        """
        B = pose.size(0)
        pose_flat = pose.view(B, -1)
        mu, log_var = self.encode(pose_flat, body)
        z = self.reparameterize(mu, log_var)
        pose_recon = self.decode(z, body).view(B, 64, 33, 3)
        return pose_recon, mu, log_var

    def anomaly_score(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """reconstruction error per sample — 이상 점수 (낮을수록 정상)
        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
        Returns:
            score: (B,)
        """
        B = pose.size(0)
        pose_flat = pose.view(B, -1)
        mu, _ = self.encode(pose_flat, body)
        pose_recon = self.decode(mu, body)
        return F.mse_loss(pose_recon, pose_flat, reduction='none').mean(dim=1)

    def joint_attribution(self, pose: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """관절별 재구성 오류 → heatmap.

        BC-STNF의 gradient 기반 방식과 달리 직접적: 어느 관절이 잘못 재구성됐는지 바로 확인.

        Args:
            pose: (B, 64, 33, 3)
            body: (B, 7)
        Returns:
            heatmap: (B, 64, 33) — 값이 클수록 해당 관절/프레임 재구성 오류 큼
        """
        with torch.no_grad():
            B = pose.size(0)
            pose_flat = pose.view(B, -1)
            mu, _ = self.encode(pose_flat, body)
            pose_recon = self.decode(mu, body).view(B, 64, 33, 3)
            # 관절별 유클리드 오류: (B, 64, 33, 3) → norm over xyz → (B, 64, 33)
            return (pose_recon - pose).pow(2).sum(dim=-1).sqrt()

    @staticmethod
    def compute_loss(
        pose: torch.Tensor,
        pose_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ):
        """
        Returns:
            total_loss, recon_loss, kl_loss
        """
        B = pose.size(0)
        pose_flat = pose.view(B, -1)
        recon = F.mse_loss(pose_recon.view(B, -1), pose_flat, reduction='mean')
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon + beta * kl, recon, kl

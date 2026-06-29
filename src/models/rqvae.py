import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    Single-level VQ with:
    - KMeans-warm-start initialization (optional)
    - commitment loss + codebook loss + diversity loss (LETTER 방식)
    """
    def __init__(self, n_e: int = 256, e_dim: int = 256,
                 mu: float = 0.25, beta: float = 1.0):
        super().__init__()
        self.n_e   = n_e
        self.e_dim = e_dim
        self.mu    = mu    # commitment loss weight
        self.beta  = beta  # diversity loss weight
        self.embedding = nn.Embedding(n_e, e_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / n_e, 1.0 / n_e)

    def init_from_kmeans(self, centroids: np.ndarray):
        """기존 KMeans centroid로 초기화 (Option B)."""
        assert centroids.shape == (self.n_e, self.e_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(centroids).float())

    def forward(self, z: torch.Tensor, label=None):
        """
        z: (N, e_dim)
        returns: z_q (quantized), loss, indices (N,)
        """
        # distances to all centroids
        d = (torch.sum(z ** 2, dim=1, keepdim=True)
             + torch.sum(self.embedding.weight ** 2, dim=1)
             - 2 * z @ self.embedding.weight.T)   # (N, n_e)

        indices = torch.argmin(d, dim=1)           # (N,)
        z_q = self.embedding(indices)              # (N, e_dim)

        # commitment + codebook loss
        commitment_loss = F.mse_loss(z_q.detach(), z)
        codebook_loss   = F.mse_loss(z_q, z.detach())

        # diversity loss: encourage uniform usage across centroids
        # soft assignment via negative distance
        soft = F.softmax(-d / 0.1, dim=-1)         # (N, n_e)
        avg  = soft.mean(dim=0)                     # (n_e,)
        diversity_loss = (avg * torch.log(avg + 1e-8)).sum()  # entropy (negative = maximize)

        loss = codebook_loss + self.mu * commitment_loss + self.beta * diversity_loss

        # straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, e_dim: int = 256, depth: int = 4,
                 n_e: int = 256, mu: float = 0.25, beta: float = 1.0):
        super().__init__()
        self.depth = depth
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(n_e, e_dim, mu, beta) for _ in range(depth)
        ])

    def init_from_kmeans(self, centroids_list: list):
        """각 level의 KMeans centroid로 초기화."""
        for i, vq in enumerate(self.vq_layers):
            vq.init_from_kmeans(centroids_list[i])

    def forward(self, z: torch.Tensor):
        """
        z: (N, e_dim)
        returns: z_q_sum (N, e_dim), total_loss, all_indices (N, depth)
        """
        residual = z
        z_q_sum  = torch.zeros_like(z)
        all_losses   = []
        all_indices  = []

        for vq in self.vq_layers:
            z_q, loss, indices = vq(residual)
            residual = residual - z_q.detach()
            z_q_sum  = z_q_sum + z_q
            all_losses.append(loss)
            all_indices.append(indices)

        total_loss  = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=1)  # (N, depth)
        return z_q_sum, total_loss, all_indices


class EntityRQVAE(nn.Module):
    """
    Encoder: Linear(768→256) → GELU → LayerNorm
    RQ:      ResidualVectorQuantizer(256, depth=4, n_e=256)
    Decoder: Linear(256→512) → GELU → LayerNorm → Linear(512→768)
    """
    def __init__(self, input_dim: int = 768, latent_dim: int = 256,
                 depth: int = 4, n_e: int = 256,
                 mu: float = 0.25, beta: float = 1.0, recon_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )
        self.rq = ResidualVectorQuantizer(latent_dim, depth, n_e, mu, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, input_dim),
        )

    def init_from_kmeans(self, centroids_list: list):
        self.rq.init_from_kmeans(centroids_list)

    def forward(self, x: torch.Tensor):
        """
        x: (N, 768) frozen CLIP emb
        returns: loss, indices (N, depth)
        """
        z     = self.encoder(x)                        # (N, 256)
        z_q, rq_loss, indices = self.rq(z)             # (N, 256), scalar, (N, 4)
        x_hat = self.decoder(z_q)                      # (N, 768)

        recon_loss = F.mse_loss(x_hat, x)
        total_loss = self.recon_weight * recon_loss + rq_loss

        return total_loss, recon_loss, rq_loss, indices

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """추론 시 코드만 추출."""
        z = self.encoder(x)
        _, _, indices = self.rq(z)
        return indices  # (N, depth)

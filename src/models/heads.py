import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, out_dim),
    )


class QueryEntityHead(nn.Module):
    """h_q_ent: query image + question → z_q_ent"""
    def __init__(self, in_dim: int = 1536, hidden_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.mlp = _build_mlp(in_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


class EntityItemHead(nn.Module):
    """g_ent: entity text + entity image → z_ent"""
    def __init__(self, in_dim: int = 1536, hidden_dim: int = 768, out_dim: int = 512):
        super().__init__()
        self.mlp = _build_mlp(in_dim, hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)

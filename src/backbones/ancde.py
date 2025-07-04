import torch
from torch import nn

from .ncde import NeuralCDE


class AttentiveNCDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, tol: float):
        super(AttentiveNCDE, self).__init__()
        self.ncde_attn = NeuralCDE(input_dim, hidden_dim, tol=tol, reduction="event")
        self.attn_proj = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.ncde_final = NeuralCDE(input_dim, hidden_dim, tol=tol)

    def forward(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor):
        attn_emb = self.ncde_attn(x, time, mask)
        attn = self.attn_proj(attn_emb)
        weighted_x = x * attn
        final = self.ncde_final(weighted_x, time, mask)
        return final

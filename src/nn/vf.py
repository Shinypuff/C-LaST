"""Module with all the considered dynamics vector fields."""

import torch
from torch import Tensor, nn

from ..interp.base import BaseInterp


class MultiHeadFeedForwardVF(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, interp: BaseInterp):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.interp = interp
        headdim = hidden_dim // nhead
        self.W1 = nn.Parameter(torch.empty(nhead, headdim * 4, headdim))
        self.W2 = nn.Parameter(torch.empty(nhead, headdim, headdim * 4))
        self.b1 = nn.Parameter(torch.empty(nhead, headdim * 4, 1))
        self.b2 = nn.Parameter(torch.empty(nhead, headdim, 1))

        self.W_proj = nn.Parameter(torch.empty(nhead, headdim * headdim, headdim))
        self.headdim = headdim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")
        nn.init.zeros_(self.b1)
        nn.init.zeros_(self.b2)

        nn.init.xavier_uniform_(self.W_proj)

    def forward(self, t: Tensor, h: Tensor):
        M = self.nhead
        H1 = self.headdim

        t = t.view(-1, M)
        x: Tensor = self.interp(t).view(-1, M, H1)

        h = h.view(-1, M, H1).unsqueeze(-1)

        # (M, H1, H1) @ (B, M, H1, 1) -> (B, M, H1, 1)
        h = (self.W1 @ h + self.b1).relu()
        h = (self.W2 @ h + self.b2).relu()

        # (M, H1 * H1, H1) @ (B, M, H1, 1) -> (B, M, H1 * H1, 1)
        h = (self.W_proj @ h).tanh()
        h = h.squeeze(-1).view(-1, M, H1, H1)

        # (B, M, H1, H1) @ (B, M, H1, 1) -> (B, M, H1, 1)
        dh = h @ x.unsqueeze(-1)

        # (B, M, H1, 1) -> (B * M, H1)
        dh = dh.squeeze(-1).view(-1, H1)
        return dh


class FeedForwardVF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, interp: BaseInterp):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interp = interp
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * input_dim),
            nn.Tanh(),
        )

    def forward(self, t: Tensor, h: Tensor):
        x: Tensor = self.interp(t)
        control_matrix = self.net(h).view(-1, self.hidden_dim, self.input_dim)
        return (control_matrix @ x.unsqueeze(-1)).squeeze(-1)

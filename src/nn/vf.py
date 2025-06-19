"""Module with all the considered dynamics vector fields."""

import torch
from torch import Tensor, nn

from ..interp.wss_module import MultiHeadSmoothingSpline


class FeedForwardVF(nn.Module):
    def __init__(self, hidden_dim: int, nhead: int, interp: MultiHeadSmoothingSpline):
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
        x: Tensor = self.interp(t)  # (B, M, H1)

        h = h.view(-1, M, H1).unsqueeze(-1)

        # (M, H1, H1) @ (B, M, H1, 1) -> (B, M, H1, 1)
        h = (self.W1 @ h + self.b1).relu()
        h = (self.W2 @ h + self.b2).relu()

        # (N, H1 * H1, H1) @ (B, M, H1, 1) -> (B, M, H1 * H1, 1)
        h = (self.W_proj @ h).tanh()
        h = h.squeeze(-1).view(-1, M, H1, H1)

        # (B, N, H1, H1) @ (B, N, H1, 1) -> (B, N, H1, 1)
        dh = h @ x.unsqueeze(-1)

        # (B, N, H1, 1) -> (B * N, H1)
        dh = dh.squeeze(-1).view(-1, H1)
        return dh

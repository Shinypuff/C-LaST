import torch
from signatory import logsignature_channels
from torch import nn
from torchcde import logsig_windows

from ..interp.nat_cub_spline import NaturalCubicSpline
from ..utils.mask_utils import masklast
from .ncde import NeuralCDE


class NeuralRDE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tol: float,
        depth: int,
        window_size: int,
    ):
        super().__init__()

        self.depth = depth
        self.window_size = window_size

        self.sign_size = logsignature_channels(input_dim, depth)
        self.ncde = NeuralCDE(self.sign_size, hidden_dim, tol)
        self.interp = NaturalCubicSpline()
        self.h0_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor):
        x = torch.where(mask.unsqueeze(-1), x, masklast(x, mask, dim=1, keepdim=True))
        x = logsig_windows(x, self.depth, self.window_size)
        B, L = x.shape[:-1]
        x = self.ncde(x, time, x.new_ones((B, L), dtype=bool))
        return x

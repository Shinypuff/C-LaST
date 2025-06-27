import torch
from signatory import signature_channels
from torch import nn
from torchcde import logsig_windows

from ..interp.nat_cub_spline import NaturalCubicSpline
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
        self.depth = depth
        self.window_size = window_size

        self.sign_size = signature_channels(input_dim, depth)
        self.ncde = NeuralCDE(self.sign_size, hidden_dim, tol)
        self.interp = NaturalCubicSpline()
        self.h0_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor):
        x = logsig_windows(x, self.depth, self.window_size)
        x = self.ncde(x, time, mask)
        return x

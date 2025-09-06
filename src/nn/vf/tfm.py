"""Module with all the considered dynamics vector fields."""

from torch import Tensor, nn
import torch


class MLPVF(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, t: Tensor, y: Tensor, h: Tensor):
        flow_in = torch.cat([t.unsqueeze(-1), y, h], dim=-1)
        return self.net(flow_in)

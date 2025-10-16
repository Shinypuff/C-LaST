import torch
from torch import Tensor, nn

from ...layers.mlp import MLP


class MLPVF(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = MLP(hidden_dim + 1, hidden_dim, hidden_dim)

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        input_tensor = torch.cat([x, t.unsqueeze(-1)], dim=-1)
        return self.net(input_tensor)

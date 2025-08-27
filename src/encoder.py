import torch
from tensordict import TensorDict
from torch import nn


class TSEncoder(nn.Module):
    def __init__(self, num_feats: list[str]):
        super().__init__()

        self.num_feats = sorted(num_feats)
        self.feats = self.num_feats
        self.input_dim = len(num_feats)

    def forward(self, x: TensorDict):
        x = torch.cat([x[f].unsqueeze(-1) for f in self.num_feats], dim=-1)
        return x.float()

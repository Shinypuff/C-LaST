"""Module with the vector fields for TFM."""

import torch
from torch import Tensor, nn


class DualTargetVF(nn.Module):
    """The vector field, wrapping two target-predicting models."""

    def __init__(self, mean_net: nn.Module, noise_net: nn.Module, eps: float = 1e-3):
        """Initialize the VF object."""
        super().__init__()
        self.mean_net = mean_net
        self.noise_net = noise_net
        self.eps = eps

    def forward(self, t: Tensor, dual_y: Tensor, h: Tensor):
        """Calculate the flow at the given point."""
        tu = t.unsqueeze(-1)
        time_till_next = (1 - tu).clip(min=self.eps)

        mean = dual_y[:, : dual_y.shape[1] // 2]
        flow_in = torch.cat([tu, mean, h], dim=-1)
        mean_target = self.mean_net(flow_in)
        noise_target = self.noise_net(flow_in)
        target = torch.cat([mean_target, noise_target], dim=-1)
        flow = (target - dual_y) / time_till_next
        return flow

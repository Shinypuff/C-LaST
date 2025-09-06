"""File with the Mamba2 model.

Taken from examples from the Mamba repository.
https://github.com/state-spaces/mamba
"""

import torch
from mamba_ssm.modules.mamba2 import Mamba2
from torch import Tensor, nn

from ..utils.mask_utils import masklast


class Mamba(nn.Module):
    """The Mamba2 model."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_layer: int,
        nhead: int,
    ) -> None:
        """Initialize the Mamba2 model."""
        super().__init__()

        self.mamba = nn.Sequential()

        for i in range(n_layer):
            layer = Mamba2(
                d_model=hidden_dim,
                headdim=hidden_dim // nhead,
                d_state=16,
                d_conv=4,
                expand=2,
            )

            self.mamba.add_module(f"layer_{i}", layer)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(input_dim, self.hidden_dim)

    def forward(self, embedding: Tensor, time: Tensor, mask: Tensor):
        """Forward pass of the Mamba2 model."""
        embedding = self.proj(embedding)
        embedding = self.mamba(embedding.contiguous()).contiguous()
        embedding = masklast(embedding, mask, 1)

        return embedding

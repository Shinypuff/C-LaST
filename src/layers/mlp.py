"""File with a basic MLP class."""

from typing import Callable

import torch
from torch import nn


class MLP(nn.Module):
    """A Multi-Layer Perceptron (MLP) neural network module.

    This class implements a fully connected neural network with configurable input, hidden, and output dimensions.
    The network consists of three linear layers with Batch Normalization and SELU activation functions.

    Args:
        input_dim (int): The dimension of the input features.
        hidden_dim (int): The dimension of the hidden layers.
        output_dim (int): The dimension of the output.

    Attributes:
        net (nn.Sequential): The sequential container holding the network layers.

    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_layers: int = 2,
        final_act: Callable | None = None,
    ):
        """Initialize the MLP with specified input, hidden, and output dimensions.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the output layer.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.final_act = final_act

        self.net = nn.Sequential()

        self.net.append(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SELU(),
            )
        )
        for _ in range(hidden_layers - 1):
            self.net.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SELU(),
                )
            )

        self.net.append(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.SELU(),
                nn.Linear(hidden_dim, output_dim),
            )
        )

    def forward(self, x: torch.Tensor):
        """Run the forward pass.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.

        """
        preshape = x.shape[:-1]
        x = x.reshape(-1, self.input_dim)
        out = self.net(x)
        out = out.reshape(*preshape, self.output_dim)
        if self.final_act is not None:
            out = self.final_act(out)
        return out

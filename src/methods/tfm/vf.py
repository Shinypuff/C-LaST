"""Module with the vector fields for TFM."""

import torch
from torch import Tensor, nn

from ...layers.mlp import MLP


class MLPVF(nn.Module):
    """A Multi-Layer Perceptron (MLP) based vector field model.

    This class defines a neural network module that implements a vector field using an MLP architecture.
    The vector field takes as input a time component, a state vector, and a context vector, concatenates them,
    and passes the concatenated input through an MLP to produce the output.

    Args:
        input_dim (int): The dimensionality of the input features (including time, state, and context).
        hidden_dim (int): The dimensionality of the hidden layers in the MLP.
        output_dim (int): The dimensionality of the output vector field.

    Attributes:
            net (MLP): The underlying MLP network used to compute the vector field.

    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize the VF object with specified dimensions and creates an MLP network.

        Args:
            input_dim (int): The dimension of the input features.
            hidden_dim (int): The dimension of the hidden layers in the MLP.
            output_dim (int): The dimension of the output features.

        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = MLP(input_dim, hidden_dim, output_dim)

    def forward(self, t: Tensor, y: Tensor, h: Tensor):
        """Perform a forward pass through the neural network.

        This method concatenates the input tensors `t`, `y`, and `h` along the last dimension,
        adds a new axis to `t` before concatenation, and passes the resulting tensor through
        the neural network `self.net`.

        Args:
            t (Tensor): A tensor representing time, expected to be unsqueezed before concatenation.
            y (Tensor): A tensor representing the state or output from the previous step.
            h (Tensor): A tensor representing additional input features or context.

        Returns:
            Tensor: The output of the neural network after processing the concatenated input.

        """
        flow_in = torch.cat([t.unsqueeze(-1), y, h], dim=-1)
        return self.net(flow_in)

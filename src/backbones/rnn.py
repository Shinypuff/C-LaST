"""The file with the RNN backbone."""

from typing import Literal
import torch
import torchode as to
from torch import nn

from ..interp.wss_module import MultiHeadSmoothingSpline
from ..mask_utils import masklast
from ..nn.vf import FeedForwardVF


class RNN(nn.Module):
    """The RNN backbone."""

    def __init__(self, hidden_dim: int, type: Literal["rnn", "lstm", "gru"]):
        """Initialize the RNN backbone.

        Args:
        ----
            hidden_dim (int): The hidden dimension.

        """
        super().__init__()
        self.hidden_dim = hidden_dim
        match type:
            case "rnn":
                rnn_cls = nn.RNN
            case "lstm":
                rnn_cls = nn.LSTM
            case "gru":
                rnn_cls = nn.GRU
            case a:
                raise ValueError(f"Unknown RNN type: {a}")

        self.rnn = rnn_cls(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor):
        """Forward pass of the RNN backbone.

        Args:
        ----
            x (torch.Tensor): The input tensor.
            time (torch.Tensor): The time tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        h = self.rnn(x)[0]
        h = masklast(h, mask, dim=1)
        return h

"""File with RNN-based vector fields."""

from abc import ABC, abstractmethod

from torch import Tensor, nn

from ...interp.base import BaseInterpolator


class RNNVFBase(nn.Module, ABC):
    """Base class for interpolator-based VFs."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        interp: BaseInterpolator,
        nf: bool,
    ):
        """Initialize the interpolator-based VF.

        Args:
        ----
            input_size (int): Input size.
            hidden_size (int): Hidden size.
            interp (BaseInterpolator): Interpolator.
            nf (bool): Whether to use negative feedback.

        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.interp = interp
        self.nf = nf
        self._setup_network()

    @abstractmethod
    def _setup_network(self):
        """Set up the dynamics network."""

    @abstractmethod
    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""


class AntiSyncVF(RNNVFBase):
    """GRU interpolator-based VF (adaptive version)."""

    def _setup_network(self):
        """Initialize GRU."""
        self.net = nn.GRUCell(self.input_size, self.hidden_size)

    def forward(self, t: Tensor, h: Tensor):
        """Forward pass of the interpolator-based VF."""
        x = self.interp(t)
        if self.nf:
            return self.net(x, -h)
        else:
            return self.net(x, h)

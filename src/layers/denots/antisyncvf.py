from torch import Tensor, nn

from ...interp.nat_cub_spline import eval_cubic_spline


class AntiSynchVF(nn.Module):
    """GRU interpolator-based VF (adaptive version)."""

    def __init__(self, input_size: int, hidden_size: int):
        """Initialize GRU."""
        super().__init__()
        self.net = nn.GRUCell(input_size, hidden_size)

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor]):
        """Forward pass of the interpolator-based VF."""
        coeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        return self.net(x, -h)

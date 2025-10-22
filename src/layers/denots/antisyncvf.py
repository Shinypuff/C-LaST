import torch
from torch import Tensor, nn

from ...interp.nat_cub_spline import eval_cubic_spline
from ..mlp import MLP


class AntiSynchVF(nn.Module):
    """GRU interpolator-based VF (adaptive version)."""

    def __init__(self, input_size: int, hidden_size: int, sigma: float):
        """Initialize GRU."""
        super().__init__()
        self.net = nn.GRUCell(input_size, hidden_size)
        assert sigma >= 0
        if sigma:
            self.diffnet = MLP(
                hidden_size, hidden_size, hidden_size, final_act=torch.exp
            )

        self.sigma = sigma

    def forward(self, t: Tensor, h: Tensor, args: tuple[Tensor, Tensor]):
        """Forward pass of the interpolator-based VF."""
        coeffs, tobs = args
        x = eval_cubic_spline(coeffs, tobs, t)
        dh = self.net(x, -h)
        if self.sigma:
            dw = torch.randn_like(dh) * self.diffnet(h) * self.sigma
            dh = dh + dw

        return dh

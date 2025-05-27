import torch
from torch import nn

from .mask_utils import maskroll
from .natural_cubic_spline import eval_cubic_spline_1d, fit_cubic_spline_1d
from .smoothing_splines import weighted_smoothing_spline_1d


class WeightedSmoothingSpline(nn.Module):
    def __init__(self, hidden_size: int, nhead: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead  # referred to as N in comments

        self.W = nn.Parameter(torch.rand(nhead, hidden_size))

        # V-Map 1D functions to batch, hidden and head dimensions
        # head (weights) - dim 0
        wsm_n = torch.vmap(weighted_smoothing_spline_1d, (None, None, None, 0, None), 0)
        # batch - dim 1
        wsm_nb = torch.vmap(wsm_n, (0, 0, 0, 1, None), 1)
        # hidden - last dim
        self.wsm_nbh = torch.vmap(wsm_nb, (None, -1, None, None, None), -1)

        fitspline_n = torch.vmap(fit_cubic_spline_1d, (None, 0))
        fitspline_nb = torch.vmap(fitspline_n, (0, 1), 1)
        self.fitspline_nbh = torch.vmap(fitspline_nb, (None, -1), -1)

        evlspline_n = torch.vmap(eval_cubic_spline_1d, (0, None, None), 0)
        evlspline_nb = torch.vmap(evlspline_n, (1, 0, 0), 1)
        self.evlspline_nbh = torch.vmap(evlspline_nb, (-1, None, None), -1)

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        # Calculate weights
        weights = torch.inner(self.W, y)

        # Calculate knots
        knot_mask = torch.empty_like(x, dtype=bool)
        knot_mask[:, :-1] = x[:, :-1] != x[:, 1:]
        knot_mask[:, -1] = True
        new_mask, knots = maskroll(knot_mask, x)
        knots = torch.where(new_mask, knots, torch.inf)

        # Calculate the weighted smoothing spline at knots
        # (N, B, L, H)
        y_hat = self.wsm_nbh(x, y, knots, weights, 1.0)

        # Fit an interpolating spline to knots
        # (N, B, 4, L, H)
        self.coeffs = self.fitspline_nbh(knots, y_hat)
        self.knots = knots

    def forward(self, x_eval):
        return self.evlspline_nbh(self.coeffs, self.knots, x_eval)

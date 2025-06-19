import torch
from torch import nn

from ..mask_utils import maskroll
from .natural_cubic_spline import eval_piecewise_poly_1d, fit_cubic_spline_1d
from .weighted_smoothing_spline import weighted_smoothing_spline_1d


class MultiHeadSmoothingSpline(nn.Module):
    """Multi-head version of a smoothing spline for multi-dimensional time series data.

    This module extends the functionality of a single 1D smoothing spline to multiple heads, enabling parallel processing
    across hidden features. It supports operations such as fitting the spline to data and evaluating the derivative
    at arbitrary points.

    Args:
        hidden_size (int): The size of the hidden dimension.
        nhead (int): The number of attention heads (referred to as N in comments).
        smoothing (float): Smoothing parameter for the spline fitting.

    Attributes:
        hidden_size (int): Size of the hidden dimension.
        nhead (int): Number of attention heads.
        smoothing (float): Smoothing parameter used in the spline fitting.
        W (nn.Parameter): Learnable weights for each head (shape: `[nhead, hidden_size]`).
        proj (nn.Linear): Projection layer for transforming input features before spline evaluation.
        wsm (function): Weighted smoothing spline 1D operation mapped over batch, head, and hidden dimensions.
        fitspline (function): Cubic spline fitting operation mapped over batch, head, and hidden dimensions.
        evalpoly (function): Polynomial evaluation operation mapped over batch, head, and hidden dimensions.
        dcoeffs (torch.Tensor): Fitted coefficients of the derivative of the cubic splines.
        knots (torch.Tensor): Knots (breakpoints) of the fitted splines.

    Methods:
        fit(t, x): Fits the smoothing splines to input time series data.
        forward(x_eval): Evaluates the derivative of the fitted splines at given evaluation times.

    """

    def __init__(self, hidden_size: int, nhead: int, smoothing: float):
        """Initialize the WSS module for weighted smoothing splines in a multi-head attention setup.

        This constructor sets up the neural module with learnable parameters and vmap-ed
        functions for batched, head-based, and hidden dimension parallelized spline operations.

        Args:
            hidden_size (int): The size of the hidden dimension.
            nhead (int): The number of attention heads (referred to as N in comments).
            smoothing (float): The smoothing parameter used in weighted smoothing splines.

        Attributes:
            W (nn.Parameter): Learnable weights with shape (nhead, hidden_size).
            proj (nn.Linear): Linear projection layer mapping hidden_size -> hidden_size.
            wsm (callable): V-Map-ed weighted_smoothing_spline_1d function over batch, head, and hidden dimensions.
            fitspline (callable): V-Map-ed fit_cubic_spline_1d function over batch, head, and hidden dimensions.
            evalpoly (callable): V-Map-ed eval_piecewise_poly_1d function over batch, head, and hidden dimensions.

        """
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead  # referred to as N in comments
        self.smoothing = smoothing

        self.W = nn.Parameter(torch.rand(nhead, hidden_size))
        self.proj = nn.Linear(hidden_size, hidden_size)

        # V-Map 1D functions to batch, hidden and head dimensions
        wsm = weighted_smoothing_spline_1d
        wsm = torch.vmap(wsm, (0, 0, 0, 0, None), 0)  # batch
        wsm = torch.vmap(wsm, (None, 1, None, 1, None), 1)  # head
        self.wsm = torch.vmap(wsm, (None, -1, None, None, None), -1)  # hidden

        fitspline = fit_cubic_spline_1d
        fitspline = torch.vmap(fitspline)  # batch
        fitspline = torch.vmap(fitspline, (None, 1), 1)  # head
        self.fitspline = torch.vmap(fitspline, (None, -1), -1)  # hidden

        evalpoly = eval_piecewise_poly_1d
        evalpoly = torch.vmap(evalpoly)  # batch
        evalpoly = torch.vmap(evalpoly, (1, None, None), 1)  # head
        self.evalpoly = torch.vmap(evalpoly, (-1, None, None), -1)  # hidden

    def fit(self, t: torch.Tensor, x: torch.Tensor):
        """Fit the wavelet smoothing spline model to the input data.

        This method computes the weighted smoothing spline at the identified knots using the input time tensor `t` and data tensor `x`.
        It first normalizes the weight matrix, calculates weights based on the inner product, identifies knots from the time tensor,
        projects the data, and fits the interpolating splines at those knots. The resulting coefficients and knots are stored in
        instance variables.

        Args:
            t (torch.Tensor): A tensor of time points with shape (B, L).
            x (torch.Tensor): A tensor of input data with shape (B, L, H).

        Attributes:
            dcoeffs (torch.Tensor): Coefficients of the fitted interpolation splines, excluding the first one.
            knots (torch.Tensor): The identified knots from the time tensor used in the fitted model.

        """
        B, L, H = x.shape
        N = self.nhead
        H1 = H // N

        # Calculate weights
        W_norm = self.W / torch.linalg.norm(self.W, dim=1, keepdim=True)

        weights = (torch.inner(W_norm, x).transpose(0, 1) + 1) / 2  # (B, N, L)

        # Calculate knots
        knot_mask = t.isfinite()
        knot_mask[:, :-1] &= t[:, :-1] != t[:, 1:]
        new_mask, knots = maskroll(knot_mask, t)
        knots = torch.where(new_mask, knots, torch.inf)

        # Calculate the weighted smoothing spline at knots
        x = self.proj(x).reshape(B, L, N, H1).transpose(1, 2)
        # Same shape as x
        x_hat = self.wsm(t, x, knots, weights, self.smoothing)

        # Fit an interpolating spline to knots
        # (B, N, 4, L, H1)
        coeffs: torch.Tensor = self.fitspline(knots, x_hat)
        self.dcoeffs = coeffs[:, :, 1:]
        self.knots = knots

    def forward(self, x_eval):
        """Evaluate the spline's derivative at the given batch of times.

        Args:
            x_eval: batch of evaluation times, (B).

        Returns:
            Value of derivative of each spline head at x_eval,
            shape (B, N, H)

        """
        return self.evalpoly(self.dcoeffs, self.knots, x_eval)

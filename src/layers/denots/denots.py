"""File with the DeNOTS backbone class."""

import torch
import torchode as to
from torch import Tensor, nn

from ...interp.nat_cub_spline import fit_cubic_spline
from .antisyncvf import AntiSynchVF


class DeNOTS(nn.Module):
    """The DeNOTS backbone."""

    def __init__(
        self,
        input_dim,
        hidden_dim: int,
        timescale: float = 10,
    ):
        """Initialize the DeNOTS backbone.

        Args:
        ----
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            interp (BaseInterpolator): The interpolator.
            vf_type (str): The type of vector field (adaptive or strict).
            nf (bool): Whether to use negative feedback.
            timescale (float): The "timescale" of the ODE.
            tol (float): The tolerance of the ODE solver.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.timescale = timescale

        term = to.ODETerm(AntiSynchVF(input_dim, self.hidden_dim), with_args=True)
        stepper = to.Dopri5(term)
        controller = to.IntegralController(1e-3, 1e-3, term=term)
        self.solver = to.AutoDiffAdjoint(
            stepper,
            controller,
            backprop_through_step_size_control=False,
        )

    def forward(
        self, x: Tensor, t: Tensor, y0: Tensor | None = None, return_intermediate=False
    ):
        """Integrate the underlying ODE & apply post-processing."""
        B, L = x.shape[:-1]

        t = (t - t[:, 0, None]) * (self.timescale / L)
        coeffs = fit_cubic_spline(t, x)

        if y0 is None:
            y0 = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

        if not return_intermediate:
            ivp = to.InitialValueProblem(y0, t_start=t[:, 0], t_end=t[:, -1])  # type: ignore
            solution: to.Solution = self.solver.solve(ivp, args=(coeffs, t))
            return solution.ys[:, -1]
        else:
            ivp = to.InitialValueProblem(y0, t_start=t[:, 0], t_end=t[:, -1], t_eval=t)  # type: ignore

            solution: to.Solution = self.solver.solve(ivp, args=(coeffs, t))
            return solution.ys

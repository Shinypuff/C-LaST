"""File with the DeNOTS backbone class."""

import torch
import torchode as to
from torch import Tensor, nn

from ..interp.nat_cub_spline import NaturalCubicSpline
from ..nn.vf.rnn import AntiSyncVF
from ..utils.mask_utils import maskmax


class DeNOTS(nn.Module):
    """The DeNOTS backbone."""

    def __init__(
        self,
        input_dim,
        hidden_dim: int,
        tol: float = 1e-3,
    ):
        """Initialize the DeNOTS backbone.

        Args:
        ----
            input_dim (int): The input dimension.
            hidden_dim (int): The hidden dimension.
            vf_type (str): The type of vector field (adaptive or strict).
            nf (bool): Whether to use negative feedback.
            depth (float): The "depth" of the ODE.
            tol (float): The tolerance of the ODE solver.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.interp = NaturalCubicSpline()
        self.tol = tol

        f = AntiSyncVF(input_dim, self.hidden_dim, interp=self.interp, nf=self.nf)
        term = to.ODETerm(f)
        stepper = to.Dopri5(term)
        controller = to.IntegralController(tol, tol, term=term)
        self.solver = to.AutoDiffAdjoint(
            stepper,
            controller,
            backprop_through_step_size_control=False,
        )

    def forward(self, embedding: Tensor, time: Tensor, mask: Tensor):
        """Integrate the underlying ODE & apply post-processing."""
        B = embedding.shape[0]

        y0 = torch.zeros(
            B, self.hidden_dim, device=embedding.device, dtype=embedding.dtype
        )

        # Find max time
        tmax = maskmax(time, mask, 1)

        # make time infinite at padding
        # to not interpolate it.
        time[~mask] = torch.inf
        self.interp.fit(time, embedding)

        t_start = torch.zeros(B, dtype=time.dtype, device=time.device)
        ivp = to.InitialValueProblem(y0, t_start=t_start, t_end=tmax)  # type: ignore
        solution: to.Solution = self.solver.solve(ivp, term=self.term)
        return solution.ys[:, -1]

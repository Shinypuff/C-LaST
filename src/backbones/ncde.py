import torch
import torchode as to
from torch import nn

from ..interp.nat_cub_spline import NaturalCubicSpline
from ..nn.vf import FeedForwardVF
from ..utils.mask_utils import masklast


# TODO: the coeffs can be precomputed if this takes too much time.
class NeuralCDE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, tol: float, reduction="last"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tol = tol
        self.reduction = reduction

        self.interp = NaturalCubicSpline()
        self.h0_proj = nn.Linear(input_dim, hidden_dim)

        self.f = FeedForwardVF(input_dim, hidden_dim, interp=self.interp)
        self.term = to.ODETerm(self.f)
        self.stepper = to.Dopri5(self.term)
        self.controller = to.IntegralController(self.tol, self.tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            step_method=self.stepper,
            step_size_controller=self.controller,
            backprop_through_step_size_control=False,
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor, mask: torch.Tensor):
        """Forward pass of the NCDEFormer backbone.

        Args:
        ----
            x (torch.Tensor): The input tensor (B, L, H).
            time (torch.Tensor): The time tensor.
            mask (torch.Tensor): The mask tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        B, L = x.shape[:-1]

        regtime = torch.arange(0, L, device=x.device, dtype=x.dtype)
        regtime = regtime.broadcast_to((B, L))

        self.interp.fit(regtime, x)

        h0 = self.h0_proj(x[:, 0])
        t_start = regtime.new_zeros(B)
        t_end = masklast(regtime, mask, dim=1)

        match self.reduction:
            case "last":
                ivp = to.InitialValueProblem(h0, t_start=t_start, t_end=t_end)
                solution: to.Solution = self.solver.solve(ivp, term=self.term)

                embedding = solution.ys[:, -1]
            case "event":
                ivp = to.InitialValueProblem(
                    h0, t_start=t_start, t_end=t_end, t_eval=regtime
                )
                solution: to.Solution = self.solver.solve(ivp, term=self.term)

                embedding = solution.ys
        return embedding

import torch
import torchode as to
from torch import nn

from ..interp.nat_cub_spline import NaturalCubicSpline
from ..nn.vf import MultiHeadVF
from ..utils.mask_utils import masklast


class MultiHeadNeuralCDE(nn.Module):
    """MultiHead (Block-Diagonal) NCDE Class.

    Notes:
        This is used as a simple NCDE for EvS, with nhead=1, since
        for EvS input_dim=hidden_dim, so no projection happens.

    """

    def __init__(self, input_dim: int, hidden_dim: int, nhead: int, tol: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.headdim = hidden_dim // nhead
        self.tol = tol

        self.interp = NaturalCubicSpline()
        self.h0_proj = nn.Linear(input_dim, hidden_dim)

        if input_dim != hidden_dim:
            self.x_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.x_proj = nn.Identity()

        self.f = MultiHeadVF(hidden_dim, nhead=nhead, interp=self.interp)
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
        M = self.nhead
        H1 = self.headdim

        h0 = self.h0_proj(x[:, 0]).view(B * M, H1)
        x = self.x_proj(x)

        regtime = torch.arange(0, L, device=x.device, dtype=x.dtype)
        regtime = regtime.broadcast_to((B, L))

        self.interp.fit(regtime, x.view(B, M, L, H1))

        t_start = regtime.new_zeros(B * M)
        t_end = masklast(regtime, mask, dim=1).unsqueeze(-1).expand(B, M).flatten()

        ivp = to.InitialValueProblem(h0, t_start=t_start, t_end=t_end)
        solution: to.Solution = self.solver.solve(ivp, term=self.term)

        embedding = solution.ys[:, -1].view(B, -1)
        return embedding

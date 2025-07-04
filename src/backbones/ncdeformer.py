"""The file with the NCDEFormer backbone."""

from math import sqrt

import torch
import torchode as to
from torch import nn

from ..interp.kernel_regression import KernelRegression
from ..nn.vf import MultiHeadFeedForwardVF
from ..utils.mask_utils import maskmax


class NCDEFormer(nn.Module):
    """The NCDEFormer backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        nhead: int,
        bandwidth: float = 1.0,
        temperature: float = 1.0,
        tol: float = 1e-3,
        disable_weights=False,
    ):
        """Initialize the NCDEFormer backbone.

        Args:
        ----
            hidden_dim (int): The hidden dimension.
            nhead (int): the number of heads.
            smoothing (float): the smoothing value.
            tol (float): The tolerance of the ODE solver.

        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  # referred to as H
        self.nhead = nhead  # referred to as M
        self.tol = tol
        self.disable_weights = disable_weights

        self.interp = KernelRegression(
            bandwidth=bandwidth,
            temperature=temperature,
        )

        self.Q = nn.Parameter(torch.empty(nhead, hidden_dim))
        self.k_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.h0_proj = nn.Linear(input_dim, hidden_dim)
        self.headdim = hidden_dim // nhead
        self.f = MultiHeadFeedForwardVF(hidden_dim, nhead=nhead, interp=self.interp)
        self.term = to.ODETerm(self.f)
        self.stepper = to.Dopri5(self.term)
        self.controller = to.IntegralController(self.tol, self.tol, term=self.term)
        self.solver = to.AutoDiffAdjoint(
            step_method=self.stepper,
            step_size_controller=self.controller,
            backprop_through_step_size_control=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the NCDEFormer backbone."""
        nn.init.xavier_uniform_(self.Q)

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
        H = self.hidden_dim
        M = self.nhead
        H1 = self.headdim

        keys = self.k_proj(x)  # (B, L, H)

        if self.disable_weights:
            weights = torch.zeros(B, M, L, device=x.device)
        else:
            # Einsum is not the fastest way to do this.
            # (but the most explicit, and we do this only once)
            weights = torch.einsum("mh,blh->bml", self.Q, keys) / sqrt(H)  # (B, M, L)

        values = self.v_proj(x).view(B, L, M, H1).transpose(1, 2)  # (B, M, L, H1)

        regtime = torch.arange(0, L, device=x.device, dtype=x.dtype)
        regtime = regtime.broadcast_to((B, L))

        self.interp.fit(regtime, values, weights)

        t_start = regtime.new_zeros(B * M)
        t_end = maskmax(regtime, mask, dim=1).unsqueeze(-1).expand(B, M).flatten()
        h0 = self.h0_proj(x[:, 0]).reshape(B * M, H1)
        ivp = to.InitialValueProblem(h0, t_start=t_start, t_end=t_end)
        solution: to.Solution = self.solver.solve(ivp, term=self.term)

        embedding = solution.ys[:, -1].reshape(B, H)
        return embedding

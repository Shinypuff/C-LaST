"""The Trajectory Flow Matching approach from https://arxiv.org/abs/2410.21154."""

import torch
import torchode as to
from torch import Tensor
from torch.nn.functional import mse_loss

from ..base import BaseForecasting
from .vf import MLPVF


class TrajectoryFlowMatching(BaseForecasting):
    def __init__(
        self,
        history: int,
        # sigma: float,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        # self.sigma = sigma
        self.history = history

        input_dim = history * (self.ctx_dim + self.tgt_dim) + self.tgt_dim + 1
        self.flow = MLPVF(input_dim, self.hidden_dim, self.tgt_dim)

        term = to.ODETerm(self.flow, with_args=True)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-4, rtol=1e-4, term=term)
        self.solver = to.AutoDiffAdjoint(
            step_method, step_size_controller, backprop_through_step_size_control=False
        )

        # Prediction takes a while, so we monitor val loss
        self.monitor_name = "val_mse_loss"
        self.monitor_mode = "min"

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        batch = self._scale(*batch)
        loss = self.calc_loss(*batch)
        self.log("train_mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        batch = self._scale(*batch)
        loss = self.calc_loss(*batch)
        self.log("val_mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def calc_loss(self, ctx, obs, tgt):
        y = torch.cat([obs, tgt], dim=1)
        X = torch.cat([ctx, y], dim=-1)

        B, T = X.shape[:-1]
        H = self.history
        device = ctx.device

        k = torch.randint(H + 1, T - 1, (B,), device=device)
        t = torch.rand(B, device=device)
        y_k = y[torch.arange(B), k]
        y_kp1 = y[torch.arange(B), k + 1]

        y_t = (1 - t)[:, None] * y_k + t[:, None] * y_kp1
        # sigma_t = torch.sqrt((self.sigma**2) * t * (1 - t))
        # y_t = torch.randn(B, device=device) * sigma_t + mu_t
        dy_t = y_kp1 - y_k

        hist_idx = k[:, None] - torch.arange(H, 0, -1, device=device)[None, :]
        history = X[torch.arange(B)[:, None], hist_idx]

        u_t = self.flow(t, y_t, history.flatten(1, -1))
        loss = mse_loss(u_t, dy_t)
        return loss

    def forward(self, ctx: Tensor, obs: Tensor, T: int):
        B = ctx.shape[0]
        L = obs.shape[1]
        H = self.history
        y_history = obs[:, -H:]
        preds = []

        for k in range(L, L + T):
            h = torch.cat([y_history, ctx[:, k - H : k]], dim=-1).flatten(1, -1)
            y_k = y_history[:, -1]
            ivp = to.InitialValueProblem(y_k, y_k.new_zeros(B), y_k.new_ones(B))
            solution = self.solver.solve(ivp, args=h)
            y_kp1 = solution.ys[:, -1]
            y_history = torch.roll(y_history, -1, 1)
            y_history[:, -1] = y_kp1
            preds.append(y_kp1)

        preds = torch.stack(preds, dim=1)
        return preds, torch.zeros_like(preds)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

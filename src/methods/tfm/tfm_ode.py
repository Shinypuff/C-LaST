"""The Trajectory Flow Matching approach from https://arxiv.org/abs/2410.21154."""

import torch
import torchode as to
from torch import Tensor
from torch.nn.functional import mse_loss

from ...layers.mlp import MLP
from ..base import BaseForecasting
from .dual_target_vf import DualTargetVF


class TrajectoryFlowMatchingODE(BaseForecasting):
    """TrajectoryFlowMatching is a forecasting model based on flow matching techniques.

    This class implements a trajectory-based flow matching approach for time series forecasting.
    It uses a neural ODE solver to model the dynamics of the time series and predict future values.
    The model is trained to minimize the mean squared error between predicted and actual trajectories.

    Attributes:
        history (int): The number of historical time steps to consider for prediction.
        flow (MLPVF): The neural network modeling the vector field for the ODE.
        solver (to.AutoDiffAdjoint): The ODE solver with adjoint sensitivity analysis.
        monitor_name (str): The name of the metric to monitor during training ("val_mse_loss").
        monitor_mode (str): The mode for monitoring the metric ("min").

    Methods:
        training_step: Performs a single training step, computing and logging the training loss.
        validation_step: Performs a single validation step, computing and logging the validation loss.
        calc_loss: Computes the loss for a given batch using flow matching.
        forward: Generates predictions for a given context and observation sequence.
        configure_optimizers: Configures the optimizer for training.

    """

    def __init__(
        self,
        history: int,
        sigma: float,
        hidden_dim: int,
        lr: float,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        input_dim = (history + 1) * (self.ctx_dim + self.tgt_dim) + self.tgt_dim + 1
        self.mean_mlp = MLP(
            input_dim,
            hidden_dim,
            self.tgt_dim,
            hidden_layers=2,
            final_act=None,
        )

        self.noise_mlp = MLP(
            input_dim,
            hidden_dim,
            self.tgt_dim,
            hidden_layers=2,
            final_act=torch.exp,
        )

        dual_target_vf = to.ODETerm(
            DualTargetVF(self.mean_mlp, self.noise_mlp), with_args=True
        )

        self.solver = to.AutoDiffAdjoint(
            to.Dopri5(term=dual_target_vf),
            to.IntegralController(atol=1e-4, rtol=1e-4, term=dual_target_vf),
        )

        self.sigma = sigma
        self.history = history
        self.hidden_dim = hidden_dim
        self.lr = lr

    def calc_loss(self, ctx, obs, tgt):
        y = torch.cat([obs, tgt], dim=1)

        # Shift context 1 into the future
        X = torch.cat([ctx[:, 1:], y[:, :-1]], dim=-1)
        time = torch.cumsum(ctx[..., 0], dim=1)

        B, T = X.shape[:-1]
        H = self.history
        device = ctx.device

        k = torch.randint(H + 1, T - 1, (B,), device=device)
        t_01 = torch.rand(B, device=device)

        t_k = time[torch.arange(B), k]
        t_kp1 = time[torch.arange(B), k + 1]
        y_k = y[torch.arange(B), k]
        y_kp1 = y[torch.arange(B), k + 1]

        # Construct history, such that
        # history[:, 0] is time till next
        # so it's reversed (0, -1, -2, ...)
        hist_idx = k[:, None] - torch.arange(0, H + 1, device=device)[None, :]
        history = X[torch.arange(B)[:, None], hist_idx].flatten(1, -1)

        tu_01 = t_01.unsqueeze(-1)
        mu_t = (1 - tu_01) * y_k + tu_01 * y_kp1
        y_t = mu_t + self.sigma * torch.randn(B, self.tgt_dim, device=device)
        t = t_k + t_01 * (t_kp1 - t_k)
        input_tensor = torch.cat([history, y_t, t[:, None]], dim=-1)

        yhat = self.mean_mlp(input_tensor)
        noise = self.noise_mlp(input_tensor)

        mean_loss = mse_loss(y_kp1, yhat)
        uncertainty = (y_kp1 - yhat).abs()
        noise_loss = mse_loss(uncertainty.detach(), noise)
        return mean_loss + noise_loss

    def forward(self, ctx: Tensor, obs: Tensor, T: int):
        B = ctx.shape[0]
        L = obs.shape[1]
        H = self.history
        D = obs.shape[-1]

        # History is reversed (flip), 0, -1, -2, ...!
        y_history = obs[:, -(H + 1) :].flip(1)
        time = torch.cumsum(ctx[..., 0], dim=1)

        # mean + scale = dual variable
        dual_y0 = torch.cat([obs[:, -1], obs.new_zeros(B, D)], dim=-1)

        y_preds = []
        s_preds = []

        # Predict the k-th value
        for k in range(L, L + T):
            t0 = time[:, k - 1]
            t1 = time[:, k]

            # Note the history reversal here as well (flip)
            ctx_history = ctx[:, k - H : k + 1 :].flip(1)
            h = torch.cat([ctx_history, y_history], dim=-1).flatten(1, -1)

            ivp = to.InitialValueProblem(dual_y0, t0, t1)
            solution = self.solver.solve(ivp, args=(h, t1))
            dual_y0 = solution.ys[:, -1]
            y_pred, s_pred = torch.chunk(dual_y0, 2, dim=-1)

            # Insert new value !at the front! (reversed history)
            y_history[:, 1:] = y_history[:, :-1]
            y_history[:, 1] = y_pred

            y_preds.append(y_pred)
            s_preds.append(s_pred)

        y_preds = torch.stack(y_preds, dim=1)
        s_preds = torch.stack(s_preds, dim=1)

        return y_preds, s_preds

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        batch = self.scale(*batch)
        loss = self.calc_loss(*batch)
        self.log("train_mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

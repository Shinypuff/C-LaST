"""The Trajectory Flow Matching approach from https://arxiv.org/abs/2410.21154."""

import torch
import torchode as to
from torch import Tensor
from torch.nn.functional import mse_loss
from tqdm.auto import trange

from ...layers.mlp import MLP
from ..base import BaseForecasting
from .dual_target_vf import DualTargetVF


class TrajectoryFlowMatching(BaseForecasting):
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

        # Input dim =
        # + history * (time_feats + targets) -- history
        # + time_feats + targets -- last observation
        # + target + 1 -- current field value & time
        input_dim = (
            (history + 1) * (self.time_feat_dim + self.target_dim) + self.target_dim + 1
        )

        self.mean_mlp = MLP(
            input_dim,
            hidden_dim,
            self.target_dim,
            hidden_layers=2,
            final_act=None,
        )

        self.noise_mlp = MLP(
            input_dim,
            hidden_dim,
            self.target_dim,
            hidden_layers=2,
            final_act=torch.exp,
        )

        # dual_target_vf = to.ODETerm(
        #     DualTargetVF(self.mean_mlp, self.noise_mlp), with_args=True
        # )

        # self.solver = to.AutoDiffAdjoint(
        #     to.Euler(term=dual_target_vf),
        #     to.FixedStepController(term=dual_target_vf, dt0=0.2),
        # )

        self.sigma = sigma
        self.history = history
        self.hidden_dim = hidden_dim
        self.lr = lr

    def calc_loss(self, time_feats: Tensor, x: Tensor, y: Tensor):
        z = torch.cat([x, y], dim=1)
        # Shift dts 1 into the future
        all_feats = torch.cat([time_feats[:, 1:], z[:, :-1]], dim=-1)

        time = torch.cumsum(time_feats[..., 0], dim=1)

        B, T = y.shape[:-1]
        H = self.history
        device = x.device

        k = torch.randint(H + 1, T - 1, (B,), device=device)
        t_01 = torch.rand(B, device=device)

        t_k = time[torch.arange(B), k]
        t_kp1 = time[torch.arange(B), k + 1]
        z_k = z[torch.arange(B), k]
        z_kp1 = z[torch.arange(B), k + 1]

        # Construct history, such that
        # history[:, 0] is time till next
        # so it's reversed (0, -1, -2, ...)
        hist_idx = k[:, None] - torch.arange(0, H + 1, device=device)[None, :]
        history = all_feats[torch.arange(B)[:, None], hist_idx].flatten(1, -1)

        tu_01 = t_01.unsqueeze(-1)
        mu_t = (1 - tu_01) * z_k + tu_01 * z_kp1
        z_t = mu_t + self.sigma * torch.randn(B, self.target_dim, device=device)
        t = t_k + t_01 * (t_kp1 - t_k)
        input_tensor = torch.cat([history, z_t, t[:, None]], dim=-1)

        xyhat = self.mean_mlp(input_tensor)
        noise = self.noise_mlp(input_tensor)

        mean_loss = mse_loss(z_kp1, xyhat)
        uncertainty = (z_kp1 - xyhat).abs()
        noise_loss = mse_loss(uncertainty.detach(), noise)
        return mean_loss + noise_loss

    def sample(self, time_feats: Tensor, x: Tensor, num_samples: int):
        H = self.history

        # History is reversed (flip), 0, -1, -2, ...!
        y_history_0 = x[:, -(H + 1) :].flip(1)
        time = torch.cumsum(time_feats[..., 0], dim=1)

        samples = []
        for _ in trange(num_samples, desc="Sampling", leave=False):
            y_pred = x[:, -1]
            y_history = y_history_0.clone()
            preds = []

            # Predict the k-th value
            for k in range(self.context, self.context + self.horizon):
                t0 = time[:, k - 1]
                # t1 = time[:, k]

                # Note the history reversal here as well (flip)
                time_feats_history = time_feats[:, k - H : k + 1].flip(1)
                h = torch.cat([time_feats_history, y_history], dim=-1).flatten(1, -1)

                # Construct the flow's input tensor
                # history + last value + time
                flow_in = torch.cat([h, y_pred, t0.unsqueeze(-1)], dim=-1)

                # Predict next value mean & scale
                m_pred = self.mean_mlp(flow_in)
                s_pred = self.noise_mlp(flow_in)

                # Sample the prediction
                y_pred = torch.randn_like(m_pred) * s_pred + m_pred

                # Insert new value !at the front! (reversed history)
                y_history[:, 1:] = y_history[:, :-1]
                y_history[:, 1] = y_pred

                preds.append(y_pred)

            preds = torch.stack(preds, dim=1)
            samples.append(preds.to("cpu", non_blocking=True))

        return torch.stack(samples, dim=-1)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        loss = self.calc_loss(*batch)
        self.log("train_mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

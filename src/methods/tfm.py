"""The Trajectory Flow Matching approach from https://arxiv.org/abs/2410.21154."""

import torch
import torchode as to
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn.functional import mse_loss

from ..nn.vf.tfm import MLPVF


class TrajectoryFlowMatching(LightningModule):
    def __init__(
        self,
        ctx_dim: int,
        tgt_dim: int,
        hidden_dim: int,
        history: int,
        # sigma: float,
        learning_rate: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # self.sigma = sigma
        self.learning_rate = learning_rate
        self.history = history

        input_dim = (history + 1) * (ctx_dim + tgt_dim)
        self.flow = MLPVF(input_dim, hidden_dim, tgt_dim)

        term = to.ODETerm(self.flow, with_args=True)
        step_method = to.Dopri5(term=term)
        step_size_controller = to.IntegralController(atol=1e-6, rtol=1e-3, term=term)
        self.solver = to.AutoDiffAdjoint(
            step_method, step_size_controller, backprop_through_step_size_control=False
        )

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        y = torch.cat([obs, tgt], dim=1)
        X = torch.cat([ctx, y], dim=-1)

        B, T = X.shape[:-1]
        H = self.history
        device = ctx.device

        k = torch.randint(H + 1, T - 1, (B,), device=device)
        t = torch.rand(B, device=device)
        y_k = torch.gather(y, dim=1, index=k[:, None]).squeeze(1)
        y_kp1 = torch.gather(y, dim=1, index=k[:, None] + 1).squeeze(1)

        y_t = (1 - t) * y_k + t * y_kp1
        # sigma_t = torch.sqrt((self.sigma**2) * t * (1 - t))
        # y_t = torch.randn(B, device=device) * sigma_t + mu_t
        dy_t = y_kp1 - y_k

        hist_idx = k[:, None] - torch.arange(H, 0, -1, device=device).expand(B, H)
        history = torch.gather(X, dim=1, index=hist_idx)

        u_t = self.flow(t + k, y_t, history.flatten(1, -1))

        loss = mse_loss(u_t, dy_t)
        self.log("train_mse_loss", loss, on_step=True, on_epoch=True)
        return loss

    def generate(self, ctx: Tensor, obs: Tensor, T: int):
        B = ctx.shape[0]
        L = obs.shape[1]
        H = self.history
        y_history = obs[:, -H:]
        preds = []

        for k in range(L, L + T):
            h = torch.cat([y_history, ctx[k - H : k]], dim=-1).flatten(1, -1)
            y_k = y_history[:, -1]
            ivp = to.InitialValueProblem(y_k, y_k.new_zeros(B), y_k.new_ones(B))
            solution = self.solver.solve(ivp, args=h)
            y_kp1 = solution.ys[:, -1]
            y_history = torch.roll(y_history, -1, 1)
            y_history[:, -1] = y_kp1
            preds.append(y_kp1)

        return torch.stack(preds, dim=1)

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        pred = self.generate(ctx, obs, tgt.shape[1])
        loss = mse_loss(pred, tgt)
        self.log("val_mse_loss", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        pred = self.generate(ctx, obs, tgt.shape[1])
        loss = mse_loss(pred, tgt)
        self.log("test_mse_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

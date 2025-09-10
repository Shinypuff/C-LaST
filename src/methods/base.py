"""The base class for all forecasting models."""

import torch
from numpy.typing import NDArray
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection

from ..losses.nll import gaussian_nll
from ..metrics.crps import CRPS
from ..metrics.nmae import NMAE


class BaseForecasting(LightningModule):
    ctx_mean: Tensor
    ctx_scale: Tensor
    tgt_mean: Tensor
    tgt_scale: Tensor

    def __init__(
        self,
        ctx_dim: int,
        tgt_dim: int,
        ctx_mean: NDArray,
        ctx_scale: NDArray,
        tgt_mean: NDArray,
        tgt_scale: NDArray,
        hidden_dim: int,
        learning_rate: float,
    ):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.tgt_dim = tgt_dim

        self.register_buffer("ctx_mean", ctx_mean)
        self.register_buffer("ctx_scale", ctx_scale)
        self.register_buffer("tgt_mean", tgt_mean)
        self.register_buffer("tgt_scale", tgt_scale)

        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        distribution_metrics = MetricCollection({"crps": CRPS()})
        pointwise_metrics = MetricCollection(
            {"nmae": NMAE(), "mse": MeanSquaredError(num_outputs=self.tgt_dim)}
        )

        self.val_metrics_d = distribution_metrics.clone("val_")
        self.val_metrics_p = pointwise_metrics.clone("val_")
        self.test_metrics_d = distribution_metrics.clone("test_")
        self.test_metrics_p = pointwise_metrics.clone("test_")

        self.monitor_name = "val_crps"
        self.monitor_mode = "max"

    def _scale(self, ctx: Tensor, obs: Tensor, tgt: Tensor | None = None):
        ctx = (ctx - self.ctx_mean) / self.ctx_scale
        obs = (obs - self.tgt_mean) / self.tgt_scale
        if tgt is not None:
            tgt = (tgt - self.tgt_mean) / self.tgt_scale
            return ctx, obs, tgt
        else:
            return ctx, obs

    def _unscale(self, mean: Tensor, scale: Tensor):
        return mean + self.tgt_mean, scale * self.tgt_scale

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        ctx, obs, tgt = self._scale(ctx, obs, tgt)
        mean, scale = self(ctx, obs, tgt.shape[1])
        loss = gaussian_nll(tgt, mean, scale)
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        ctx, obs = self._scale(ctx, obs)
        mean, scale = self(ctx, obs, tgt.shape[1])
        mean, scale = self._unscale(mean, scale)
        self.val_metrics_d(tgt, mean, scale)
        self.val_metrics_p(tgt, mean)
        self.log_dict(self.val_metrics_p, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics_d, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        ctx, obs, tgt = batch
        ctx, obs = self._scale(ctx, obs)
        mean, scale = self(ctx, obs, tgt.shape[1])
        mean, scale = self._unscale(mean, scale)
        self.test_metrics_d(tgt, mean, scale)
        self.test_metrics_p(tgt, mean)
        self.log_dict(self.test_metrics_p, on_step=False, on_epoch=True)
        self.log_dict(self.test_metrics_d, on_step=False, on_epoch=True)

    def forward(self, ctx: Tensor, obs: Tensor, T: int):
        raise NotImplementedError("Implement the forward method!")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

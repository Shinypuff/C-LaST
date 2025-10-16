from typing import Literal

from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection

from ..losses.nll import gaussian_nll_loss
from ..metrics.crps import CRPS
from ..metrics.nmae import NMAE


class BaseForecasting(LightningModule):
    """A base class for forecasting models using PyTorch Lightning.

    This class provides a framework for implementing forecasting models.
    It includes built-in metrics for evaluating model performance.

    """

    def __init__(
        self,
        target_dim: int,
        context: int,
        horizon: int,
        time_feat_dim: int,
    ):
        """Initialize the base method."""
        super().__init__()
        self.target_dim = target_dim
        self.context = context
        self.horizon = horizon

        distribution_metrics = MetricCollection({"crps": CRPS()})
        pointwise_metrics = MetricCollection(
            {"nmae": NMAE(), "mse": MeanSquaredError()}
        )

        self.val_metrics_d = distribution_metrics.clone("val_")
        self.val_metrics_p = pointwise_metrics.clone("val_")
        self.test_metrics_d = distribution_metrics.clone("test_")
        self.test_metrics_p = pointwise_metrics.clone("test_")

        self.monitor_name = "val_crps"
        self.monitor_mode = "min"
        self.time_feat_dim = time_feat_dim

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a single training step."""
        t, x, y = batch
        mean, scale = self(t, x)
        loss = gaussian_nll_loss(y, mean, scale)
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a validation step on a batch of data."""
        t, x, y = batch
        mean, scale = self(t, x)
        self.val_metrics_d(y, mean, scale)
        self.val_metrics_p(y, mean)
        self.log_dict(self.val_metrics_d, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics_p, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a test step on a batch of data.

        Returns:
            None

        """
        t, x, y = batch
        mean, scale = self(t, x)
        self.test_metrics_d(y, mean, scale)
        self.test_metrics_p(y, mean)
        self.log_dict(self.test_metrics_d, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics_p, on_step=False, on_epoch=True, prog_bar=True)

    def forward(self, dt: Tensor, x: Tensor):
        """Run the forward pass of the model, override this method in children.

        Returns:
            tuple: A tuple containing the mean and scale tensors.

        """
        raise NotImplementedError("Implement the forward method!")

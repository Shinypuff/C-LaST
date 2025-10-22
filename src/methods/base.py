from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection

from ..metrics.crps import ContinuousRankedProbabilityScore
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
        val_samples: int,
        test_samples: int,
    ):
        """Initialize the base method."""
        super().__init__()
        self.target_dim = target_dim
        self.context = context
        self.horizon = horizon

        self.val_samples = val_samples
        self.test_samples = test_samples

        distribution_metrics = MetricCollection(
            {"crps": ContinuousRankedProbabilityScore(compute_on_cpu=True)}
        )
        pointwise_metrics = MetricCollection(
            {
                "nmae": NMAE(compute_on_cpu=True),
                "mse": MeanSquaredError(compute_on_cpu=True),
            }
        )

        self.val_metrics_d = distribution_metrics.clone("val_")
        self.val_metrics_p = pointwise_metrics.clone("val_")
        self.test_metrics_d = distribution_metrics.clone("test_")
        self.test_metrics_p = pointwise_metrics.clone("test_")

        self.monitor_name = "val_crps"
        self.monitor_mode = "min"
        self.time_feat_dim = time_feat_dim

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a validation step on a batch of data."""
        t, x, y = batch
        trajectories: Tensor = self.sample(t, x, self.val_samples)
        self.val_metrics_d(trajectories.cpu().flatten(0, -2), y.cpu().flatten())
        self.val_metrics_p(
            trajectories.cpu().flatten(),
            y.cpu().unsqueeze(-1).expand_as(trajectories).flatten(),
        )

        self.log_dict(self.val_metrics_d, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics_p, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a test step on a batch of data.

        Returns:
            None

        """
        t, x, y = batch
        trajectories: Tensor = self.sample(t, x, self.test_samples)
        self.test_metrics_d(trajectories.cpu().flatten(0, -2), y.cpu().flatten())
        self.test_metrics_p(
            trajectories.cpu().flatten(),
            y.cpu().unsqueeze(-1).expand_as(trajectories).flatten(),
        )

        self.log_dict(self.test_metrics_d, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics_p, on_step=False, on_epoch=True, prog_bar=True)

    def sample(self, t: Tensor, x: Tensor, num_samples: int) -> Tensor:
        """Sample samples from the learned distribution, override this method in children.

        Returns:
            tuple: A tuple containing the mean and scale tensors.

        """
        raise NotImplementedError("Implement the sample method!")

"""The base class for all forecasting models."""

from typing import Literal

from numpy.typing import NDArray
from pytorch_lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanSquaredError, MetricCollection

from ..losses.nll import gaussian_nll
from ..metrics.crps import CRPS
from ..metrics.nmae import NMAE


class BaseForecasting(LightningModule):
    """A base class for forecasting models using PyTorch Lightning.

    This class provides a framework for implementing forecasting models with support for data scaling,
    training, validation, and testing steps. It includes built-in metrics for evaluating model performance
    and handles the optimization setup.

    Attributes:
        ctx_mean (Tensor): The mean values for context variables used for scaling.
        ctx_scale (Tensor): The scale values for context variables used for scaling.
        tgt_mean (Tensor): The mean values for target variables used for scaling.
        tgt_scale (Tensor): The scale values for target variables used for scaling.

    Args:
        ctx_dim (int): The dimensionality of the context variables.
        tgt_dim (int): The dimensionality of the target variables.
        ctx_mean (NDArray): The mean values for context variables.
        ctx_scale (NDArray): The scale values for context variables.
        tgt_mean (NDArray): The mean values for target variables.
        tgt_scale (NDArray): The scale values for target variables.
        learning_rate (float): The learning rate for the optimizer.

    Methods:
        scale: Scales the context, observation, and target tensors using stored mean and scale values.
        unscale: Unscales the mean and scale tensors back to the original scale.
        training_step: Defines the training step for the model, including loss computation and logging.
        validation_step: Defines the validation step for the model, including metric computation and logging.
        test_step: Defines the test step for the model, including metric computation and logging.
        forward: Abstract method to be implemented by subclasses for the model's forward pass.
        configure_optimizers: Configures and returns the optimizer for training.

    """

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
        unscale_metrics: bool = False,
    ):
        """Initialize the base method.

        Args:
            ctx_dim (int): Dimension of the context variables.
            tgt_dim (int): Dimension of the target variables.
            ctx_mean (NDArray): Mean values for normalizing context variables.
            ctx_scale (NDArray): Scale values for normalizing context variables.
            tgt_mean (NDArray): Mean values for normalizing target variables.
            tgt_scale (NDArray): Scale values for normalizing target variables.
            hidden_dim (int): Dimension of the hidden layers.

        """
        super().__init__()
        self.ctx_dim = ctx_dim
        self.tgt_dim = tgt_dim

        self.register_buffer("ctx_mean", ctx_mean)
        self.register_buffer("ctx_scale", ctx_scale)
        self.register_buffer("tgt_mean", tgt_mean)
        self.register_buffer("tgt_scale", tgt_scale)

        distribution_metrics = MetricCollection({"crps": CRPS()})
        pointwise_metrics = MetricCollection(
            {"nmae": NMAE(), "mse": MeanSquaredError(num_outputs=self.tgt_dim)}
        )

        self.val_metrics_d = distribution_metrics.clone("val_")
        self.val_metrics_p = pointwise_metrics.clone("val_")
        self.test_metrics_d = distribution_metrics.clone("test_")
        self.test_metrics_p = pointwise_metrics.clone("test_")

        self.monitor_name = "val_crps"
        self.monitor_mode = "min"
        self.unscale_metrics = unscale_metrics

    def scale(
        self,
        ctx: Tensor | None = None,
        obs: Tensor | None = None,
        tgt: Tensor | None = None,
    ):
        """Scale the context, observation, and target tensors using precomputed means and scales.

        This method normalizes the input tensors by subtracting the corresponding mean and dividing by the scale.
        The context tensor is scaled using `ctx_mean` and `ctx_scale`, while the observation and target tensors
        are scaled using `tgt_mean` and `tgt_scale`.

        Args:
            ctx (Tensor | None, optional): The context tensor to be scaled.
            obs (Tensor | None, optional): The observation tensor to be scaled.
            tgt (Tensor | None, optional): The target tensor to be scaled.

        Returns:
            tuple: A tuple containing the scaled tensors.

        """
        results = []
        if ctx is not None:
            # handle time separately (dividing by mean)
            ctx[..., 0] = ctx[..., 0] / self.ctx_mean[0]

            # standard-scale the rest
            ctx[..., 1:] = (ctx[..., 1:] - self.ctx_mean[1:]) / self.ctx_scale[1:]
            results.append(ctx)

        if obs is not None:
            obs = (obs - self.tgt_mean) / self.tgt_scale
            results.append(obs)

        if tgt is not None:
            tgt = (tgt - self.tgt_mean) / self.tgt_scale
            results.append(tgt)

        return tuple(results)

    def unscale(self, mean: Tensor, scale: Tensor):
        """Unscales the given mean and scale tensors using the target mean and scale.

        Args:
            mean (Tensor): The mean tensor to be unscaled.
            scale (Tensor): The scale tensor to be unscaled.

        Returns:
            tuple: A tuple containing the unscaled mean and scale tensors.

        """
        return mean * self.tgt_scale + self.tgt_mean, scale * self.tgt_scale

    def calc_log_metrics(
        self,
        tgt_orig: Tensor,
        mean: Tensor,
        scale: Tensor,
        stage: Literal["val", "test"],
    ):
        """Calculate and log metrics for the given mean and scale tensors.

        Args:
            tgt_orig (Tensor): The original (unscaled) target tensor.
            mean (Tensor): The mean tensor.
            scale (Tensor): The scale tensor.
            stage (Literal["val", "test"]): The stage for which the metrics are being calculated.

        """
        if self.unscale_metrics:
            mean, scale = self.unscale(mean, scale)
            tgt = tgt_orig
        else:
            tgt = self.scale(tgt=tgt_orig)[0]

        metrics_d = getattr(self, f"{stage}_metrics_d")
        metrics_p = getattr(self, f"{stage}_metrics_p")

        metrics_d(tgt, mean, scale)
        metrics_p(tgt, mean)

        self.log_dict(metrics_d, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metrics_p, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a single training step.

        This method takes a batch of data, scales the context, observations, and targets,
        computes the model's predictions, calculates the Gaussian negative log-likelihood loss,
        logs the loss, and returns it.

        Args:
            batch (tuple[Tensor, Tensor, Tensor]): A tuple containing the context, observations, and targets.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The computed loss value.

        """
        ctx, obs, tgt = batch
        ctx, obs, tgt = self.scale(ctx, obs, tgt)
        mean, scale = self(ctx, obs, tgt.shape[1])
        loss = gaussian_nll(tgt, mean, scale)
        self.log("train_nll_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a validation step on a batch of data.

        This method processes a batch of data consisting of context, observations, and targets.
        It scales the context and observations, passes them through the model to get predictions,
        unscales the predictions, and then updates and logs validation metrics.

        Args:
            batch (tuple[Tensor, Tensor, Tensor]): A tuple containing context, observations, and targets.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        ctx, obs, tgt = batch
        ctx, obs = self.scale(ctx, obs)

        mean, scale = self(ctx, obs, tgt.shape[1])
        self.calc_log_metrics(tgt, mean, scale, "val")

    def test_step(self, batch: tuple[Tensor, Tensor, Tensor], *args, **kwargs):
        """Perform a test step on a batch of data.

        This method takes a batch of data consisting of context, observations, and targets,
        scales the context and observations, passes them through the model to get predictions,
        unscales the predictions, and then updates and logs test metrics.

        Args:
            batch (tuple[Tensor, Tensor, Tensor]): A tuple containing:
                - ctx (Tensor): Context tensor.
                - obs (Tensor): Observations tensor.
                - tgt (Tensor): Targets tensor.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        ctx, obs, tgt = batch
        ctx, obs, tgt = self.scale(ctx, obs, tgt)
        mean, scale = self(ctx, obs, tgt.shape[1])
        self.calc_log_metrics(tgt, mean, scale, "test")

    def forward(self, ctx: Tensor, obs: Tensor, T: int):
        """Run the forward pass of the model, override this method in children.

        This method should a context tensor and observation tensor as input,
        and returns the mean and scale tensors of the predicted distribution.

        Args:
            ctx (Tensor): Context tensor.
            obs (Tensor): Observation tensor.
            T (int): Number of timesteps in the prediction.

        Returns:
            tuple: A tuple containing the mean and scale tensors.

        """
        raise NotImplementedError("Implement the forward method!")

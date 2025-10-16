"""File with instantiation utils."""

from hydra.utils import instantiate
from omegaconf import DictConfig

from .data.datamodule import DataModule
from .methods.base import BaseForecasting


def from_config(cfg: DictConfig):
    """Instantiate a forecasting module and datamodule from a configuration.

    This function creates a DataModule using parameters from the provided configuration,
    then instantiates a forecasting method (module) with specific context and target dimensions
    and other hyperparameters derived from the configuration and datamodule properties.

    Args:
        cfg (DictConfig): A configuration object containing keys for datasource, context, horizon,
                        target_cols, time_col, batch_size, num_workers, method, hidden_dim, and learning_rate.

    Returns:
        tuple[BaseForecasting, DataModule]: A tuple containing the instantiated forecasting module
                                            and the created datamodule.

    """
    datamodule = DataModule(
        datasource=cfg["datasource"],
        context=cfg["context"],
        horizon=cfg["horizon"],
        target_cols=cfg["target_cols"],
        time_col=cfg["time_col"],
        train_batch_size=cfg["train_batch_size"],
        eval_batch_size=cfg["eval_batch_size"],
        num_workers=cfg["num_workers"],
        has_time=cfg["has_time"],
        has_date=cfg["has_date"],
    )

    # Note to (future) self:
    # Make sure we do not pass any complex objects
    # (instantiate often plays bad with them)
    module: BaseForecasting = instantiate(
        cfg["method"],
        target_dim=datamodule.target_dim,
        context=cfg["context"],
        horizon=cfg["horizon"],
        time_feat_dim=datamodule.time_feat_dim,
    )

    return module, datamodule

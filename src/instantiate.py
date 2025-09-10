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
                        tgt_cols, time_col, batch_size, num_workers, method, hidden_dim, and learning_rate.

    Returns:
        tuple[BaseForecasting, DataModule]: A tuple containing the instantiated forecasting module
                                            and the created datamodule.

    """
    datamodule = DataModule(
        datasource=cfg["datasource"],
        context=cfg["context"],
        horizon=cfg["horizon"],
        tgt_cols=cfg["tgt_cols"],
        time_col=cfg["time_col"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # Note to (future) self:
    # Make sure we do not pass any complex objects
    # (instantiate often plays bad with them)
    module: BaseForecasting = instantiate(
        cfg["method"],
        ctx_dim=datamodule.ctx_dim,
        tgt_dim=datamodule.tgt_dim,
        ctx_mean=datamodule.ctx_mean,
        ctx_scale=datamodule.ctx_scale,
        tgt_mean=datamodule.tgt_mean,
        tgt_scale=datamodule.tgt_scale,
        hidden_dim=cfg["hidden_dim"],
        learning_rate=cfg["learning_rate"],
    )

    return module, datamodule

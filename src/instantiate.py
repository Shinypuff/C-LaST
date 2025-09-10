from hydra.utils import instantiate
from omegaconf import DictConfig

from .data.datamodule import DataModule
from .methods.base import BaseForecasting


def from_config(cfg: DictConfig):
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

from omegaconf import DictConfig

from ...train_val_test import train_val_test
from .instantiate import from_config


def supervised(cfg: DictConfig):
    module, datamodule = from_config(cfg)
    train_val_test(module, datamodule, cfg, monitor="val_metric")

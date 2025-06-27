from hydra.utils import instantiate
from omegaconf import DictConfig

from ...encoder import EvSEncoder, TSEncoder
from .data import DataModule
from .module import SupervisedModule, choose_criterion


def from_config(cfg: DictConfig):
    hidden_dim = cfg["hidden_dim"]

    match cfg["sequence_type"]:
        case "evs":
            encoder = EvSEncoder(
                cat_feats=cfg["cat_feats"],
                num_feats=cfg["num_feats"],
                emb_dim=cfg["emb_dim"],
                hidden_dim=hidden_dim,
            )
        case "ts":
            encoder = TSEncoder(
                num_feats=cfg["num_feats"],
            )

    backbone = instantiate(
        cfg["backbone"], input_dim=encoder.input_dim, hidden_dim=hidden_dim
    )

    loss_fn, metric_fn, head, activation = choose_criterion(
        cfg["criterion"], hidden_dim, cfg.get("num_classes")
    )

    module = SupervisedModule(
        encoder=encoder,
        backbone=backbone,
        head=head,
        activation=activation,
        loss_fn=loss_fn,
        metric=metric_fn,
        learning_rate=cfg["learning_rate"],
    )

    datamodule = DataModule(
        datasource=cfg["datasource"],
        batch_size=cfg["batch_size"],
        balance=cfg["balance"],
        samples_per_epoch=cfg["paradigm"]["samples_per_epoch"],
        seq_feats=encoder.feats,
    )

    return module, datamodule

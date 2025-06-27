from hydra.utils import instantiate
from omegaconf import DictConfig

from ...encoder import EvSEncoder
from .data import ColesDataModule
from .module import ColesModule


def from_config(cfg: DictConfig) -> tuple[ColesModule, ColesDataModule]:
    hidden_dim = cfg["hidden_dim"]
    encoder = EvSEncoder(
        cat_feats=cfg["cat_feats"],
        num_feats=cfg["num_feats"],
        emb_dim=cfg["emb_dim"],
        hidden_dim=hidden_dim,
    )

    backbone = instantiate(
        cfg["backbone"], input_dim=encoder.input_dim, hidden_dim=hidden_dim
    )

    module = ColesModule(
        neg_samples=cfg["paradigm"]["neg_samples"],
        margin=cfg["paradigm"]["margin"],
        encoder=encoder,
        backbone=backbone,
        learning_rate=cfg["learning_rate"],
    )

    datamodule = ColesDataModule(
        datasource=cfg["datasource"],
        window_size=cfg["paradigm"]["window_size"],
        n_slices=cfg["paradigm"]["n_slices"],
        batch_size=cfg["batch_size"],
        seq_feats=encoder.feats,
        seed=cfg["seed"],
    )

    return module, datamodule

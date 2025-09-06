from hydra.utils import instantiate
from omegaconf import DictConfig


def from_config(cfg: DictConfig):
    hidden_dim = cfg["hidden_dim"]

    datamodule = instantiate(
        cfg["benchmark"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    module = instantiate(
        cfg["method"],
        input_dim=datamodule.input_dim,
        output_dim=datamodule.output_dim,
        learning_rate=cfg["learning_rate"],
        hidden_dim=hidden_dim,
    )

    return module, datamodule

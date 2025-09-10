import gc

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.instantiate import from_config
from src.train_val_test import train_val_test


@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    method = hydra_choices["method"]
    benchmark = hydra_choices["benchmark"]
    mlflow.set_experiment(f"{benchmark}_{method}")

    module, datamodule = from_config(cfg)
    train_val_test(module, datamodule, cfg, module.monitor_name, module.monitor_mode)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

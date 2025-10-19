import gc

import hydra
import mlflow
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torchinfo import summary

from src.instantiate import from_config
from src.logging import cfg2hparams


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision("medium")
    hydra_choices = HydraConfig.get().runtime.choices
    method = hydra_choices["method"]
    benchmark = hydra_choices["benchmark"]
    mlflow.set_experiment(benchmark)
    with mlflow.start_run() as run:
        mlflow.set_tag("method", method)
        dict_cfg = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(cfg2hparams(dict_cfg))
        mlflow.log_dict(dict_cfg, "config.yaml")

        module, datamodule = from_config(cfg)
        smry = summary(module)
        mlflow.log_text(str(smry), "summary.txt")
        ckpt_callback = ModelCheckpoint(
            monitor=module.monitor_name, mode=module.monitor_mode, filename="checkpoint"
        )
        es_callback = EarlyStopping(
            monitor=module.monitor_name,
            mode=module.monitor_mode,
            patience=cfg["patience"],
        )

        trainer = Trainer(
            callbacks=[es_callback, ckpt_callback],
            logger=MLFlowLogger(run_id=run.info.run_id),
            **cfg.get("trainer_args", {}),
        )

        logger.info(f"Run ID: {run.info.run_id}")
        seed_everything(cfg["seed"])

        try:
            trainer.fit(module, datamodule)
        except KeyboardInterrupt:
            mlflow.end_run("KILLED")
            # Continue to testing after keyboard interrupt
        except Exception as e:
            mlflow.log_text(str(e.__traceback__), "traceback.txt")
            mlflow.log_text(str(e), "error.txt")
            mlflow.end_run("FAILED")
            raise

        mlflow.log_artifact(ckpt_callback.best_model_path)
        module.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])
        trainer.test(module, datamodule)

    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()

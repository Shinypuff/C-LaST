import mlflow
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torchinfo import summary

from .logging import cfg2hparams


def train_val_test(
    module: LightningModule,
    datamodule: LightningDataModule,
    cfg: DictConfig,
    monitor_name: str,
    monitor_mode: str,
):
    torch.set_float32_matmul_precision("medium")
    with mlflow.start_run() as run:
        try:
            ckpt_callback = ModelCheckpoint(
                monitor=monitor_name, mode=monitor_mode, filename="checkpoint"
            )
            es_callback = EarlyStopping(
                monitor=monitor_name, mode=monitor_mode, patience=cfg["patience"]
            )

            trainer = Trainer(
                callbacks=[es_callback, ckpt_callback],
                logger=MLFlowLogger(run_id=run.info.run_id),
                **cfg.get("trainer_args", {}),
            )

            logger.info(f"Run ID: {run.info.run_id}")

            dict_cfg = OmegaConf.to_container(cfg, resolve=True)
            mlflow.log_params(cfg2hparams(dict_cfg))
            mlflow.log_dict(dict_cfg, "config.yaml")

            smry = summary(module)
            mlflow.log_text(str(smry), "summary.txt")

            seed_everything(cfg["seed"])
            trainer.fit(module, datamodule)

            mlflow.log_artifact(ckpt_callback.best_model_path)
            module.load_state_dict(
                torch.load(ckpt_callback.best_model_path)["state_dict"]
            )
            trainer.test(module, datamodule)
        except KeyboardInterrupt:
            mlflow.end_run("KILLED")
            raise
        except Exception as e:
            mlflow.log_text(str(e.__traceback__), "traceback.txt")
            mlflow.log_text(str(e), "error.txt")
            mlflow.end_run("FAILED")
            raise

    return run.info.run_id

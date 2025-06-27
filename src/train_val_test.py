import mlflow
import torch
from funcy import omit
from hydra.core.hydra_config import HydraConfig
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


def train_val_test(
    module: LightningModule,
    datamodule: LightningDataModule,
    cfg: DictConfig,
    monitor: str,
):
    benchmark_name = HydraConfig.get().runtime.choices["benchmark"]
    mlflow.set_experiment("_".join([cfg["experiment"], benchmark_name]))
    with mlflow.start_run() as run:
        ckpt_callback = ModelCheckpoint(
            monitor=monitor, mode="max", filename="checkpoint"
        )
        es_callback = EarlyStopping(
            monitor=monitor, mode="max", patience=cfg["patience"]
        )

        trainer = Trainer(
            devices=1,
            callbacks=[es_callback, ckpt_callback],
            logger=MLFlowLogger(run_id=run.info.run_id),
            **cfg.get("trainer_args", {}),
        )

        logger.info(f"Run ID: {run.info.run_id}")

        cfg_dict = OmegaConf.to_container(cfg)
        mlflow.log_params(cfg_dict)

        smry = summary(module)
        mlflow.log_text(str(smry), "summary.txt")

        seed_everything(cfg["seed"])
        try:
            trainer.fit(module, datamodule)
        except KeyboardInterrupt:
            mlflow.end_run("KILLED")
            raise
        except Exception as e:
            mlflow.log_text(str(e.__traceback__), "traceback.txt")
            mlflow.log_text(str(e), "error.txt")
            mlflow.end_run("FAILED")
            raise

        mlflow.log_artifact(ckpt_callback.best_model_path)
        module.load_state_dict(torch.load(ckpt_callback.best_model_path)["state_dict"])

        trainer.test(module, datamodule)
    return run.info.run_id

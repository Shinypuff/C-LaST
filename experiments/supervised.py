import os

import autoroot  # noqa: F401
import hydra
import mlflow
import torch
from funcy import omit
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torchinfo import summary

from src.data import Dataset
from src.trainers.supervised import SupervisedTrainer


@hydra.main(config_path="/app/config/supervised", config_name="main")
def supervised_main(cfg: DictConfig):
    trainer = SupervisedTrainer.from_config(cfg)
    ds_path: str = (
        os.environ["DATA_DIR"] + "/preprocessed/" + cfg["datasource"] + "_{}.parquet"
    )
    train_dataset = Dataset(ds_path.format("train"))
    val_dataset = Dataset(ds_path.format("val"))
    test_dataset = Dataset(ds_path.format("test"))

    benchmark_name = HydraConfig.get().runtime.choices["benchmark"]
    mlflow.set_experiment("_".join([cfg["experiment"], benchmark_name]))

    with mlflow.start_run() as run:
        logger.info(f"Run ID: {run.info.run_id}")
        cfg_dict = OmegaConf.to_container(cfg)
        mlflow.log_params(omit(cfg_dict, "device"))

        smry = summary(trainer)
        print(smry)
        mlflow.log_text(str(smry), "summary.txt")

        try:
            trainer.fit(train_dataset, val_dataset)
        except KeyboardInterrupt:
            mlflow.end_run("KILLED")
            raise
        except Exception as e:
            mlflow.log_text(str(e.__traceback__), "traceback.txt")
            mlflow.log_text(str(e), "error.txt")
            mlflow.end_run("FAILED")
            raise

        torch.save(trainer.state_dict(), "model.pth")
        mlflow.log_artifact("model.pth")
        os.unlink("model.pth")

        trainer.test(test_dataset)


if __name__ == "__main__":
    supervised_main()

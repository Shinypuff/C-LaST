import hydra
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.frames.coles.run import coles
from src.frames.supervised.run import supervised


@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    hydra_choices = HydraConfig.get().runtime.choices
    paradigm = hydra_choices["paradigm"]
    benchmark = hydra_choices["benchmark"]
    mlflow.set_experiment(f"{cfg['experiment']}_{benchmark}_{paradigm}")
    match paradigm:
        case "coles":
            coles(cfg)
        case "supervised":
            supervised(cfg)
        case _:
            raise ValueError(f"Unknown paradigm: {paradigm}")


if __name__ == "__main__":
    main()

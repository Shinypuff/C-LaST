import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src.frames.coles.run import coles
from src.frames.supervised.run import supervised


@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    hydra_cfg = HydraConfig.get()
    paradigm = hydra_cfg.runtime.choices["paradigm"]
    match paradigm:
        case "coles":
            coles(cfg)
        case "supervised":
            supervised(cfg)
        case _:
            raise ValueError(f"Unknown paradigm: {paradigm}")


if __name__ == "__main__":
    main()

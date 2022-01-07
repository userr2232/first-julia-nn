import hydra
from omegaconf import DictConfig
from partitions import create_partitions
from folds import fold_loader
from pathlib import Path
from typing import Union
from train import run_study


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "create_partition":
        create_partitions(cfg)
    elif action == "run_study":
        run_study(cfg)
    else:
        raise ValueError("Action not implemented.")

if __name__ == "__main__":
    main()
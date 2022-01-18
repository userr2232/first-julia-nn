import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Union
from src.train import run_study
from src.partitions import create_partitions


@hydra.main(config_path="conf", config_name="config")
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
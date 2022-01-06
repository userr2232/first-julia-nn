import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
from partitions import create_partitions
from folds import fold_loader
from pathlib import Path
from typing import Union

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "create_partition":
        create_partitions(cfg)
    elif action == "train":
        for fold in fold_loader(cfg):
            print("fold", fold)

if __name__ == "__main__":
    main()
import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Union
from src.train import run_study, view_study, train_w_best_params, test
from src.partitions import create_partitions
from src.geomagindices import get_indices
from src.dash_app import dash_app
import pandas as pd


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.action
    if action == "create_partition":
        create_partitions(cfg)
    elif action == "run_study":
        run_study(cfg)
    elif action == "view_study":
        view_study(cfg)
    elif action == "train_w_best_params":
        train_w_best_params(cfg)
    elif action == "test":
        test(cfg)
    elif action == "get_indices":
        hF, (ap, f10_7) = get_indices(cfg, pd.date_range('2020-01-01', '2020-01-02', freq='3H'))
        print("HF")
        print(hF)
        print("AP")
        print(ap)
        print("f10.7")
        print(f10_7)
    else:
        raise ValueError("Action not implemented.")

if __name__ == "__main__":
    main()
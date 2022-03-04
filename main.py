import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Union
from src.train import run_study, view_study, train_w_best_params, test
from src.partitions import create_partitions
from src.inference import daily_prediction, range_prediction
from src.plots import optimization_history
from src.explain import explain
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
    elif action == "daily_prediction":
        daily_prediction(cfg)
    elif action == "range_prediction":
        range_prediction(cfg)
    elif action == "explain":
        explain(cfg)
    else:
        raise ValueError("Action not implemented.")

if __name__ == "__main__":
    main()
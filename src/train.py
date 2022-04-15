import hydra
from omegaconf import DictConfig
from operator import itemgetter
import numpy as np
import torch
from torch import optim
import logging
import optuna
from numpy.typing import ArrayLike
import pyarrow as pa
from typing import Dict, Optional, Tuple
from optuna.trial import Trial
from functools import partial
from pathlib import Path

from src.folds import fold_loader, load_everything
from src.model import Model, save_jit_model, load_jit_model
from src.engine import Engine
from src.dataset import get_dataloaders


def run_training(cfg: DictConfig, fold: Tuple[pa.Table, pa.Table], params: Optional[Dict], trial: Optional[Trial] = None, save_model: bool = False, prune: bool = True) -> ArrayLike:
    epochs, device, logger_name = itemgetter("epochs", "device", "logger")(cfg.training)
    model = Model(cfg=cfg, params=params, trial=trial)
    optimizer = optim.Adam(model.parameters(), lr=params['initial_lr'])
    engine = Engine(model=model, device=device, optimizer=optimizer)
    best_loss = np.inf
    best_matrix = None
    early_stopping_iter = cfg.training.patience
    early_stopping_counter = 0
    train_table, valid_table = fold
    train_loader, valid_loader = get_dataloaders(train_table, valid_table, **cfg.model.kwargs)
    logger = logging.getLogger(logger_name)
    conf_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    for epoch in range(epochs):
        train_loss = engine.train(train_loader)
        valid_loss = engine.evaluate(valid_loader, conf_matrix)
        TP, TN, FP, FN = itemgetter('TP', 'TN', 'FP', 'FN')(conf_matrix)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        logger.info(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {valid_loss} Accuracy: {accuracy}")
        if round(valid_loss, 3) < round(best_loss, 3):
            early_stopping_counter = 0
            best_loss = valid_loss
            best_matrix = conf_matrix.copy()
            if save_model:
                save_jit_model(cfg, model)
        else:
            early_stopping_counter += 1
        if prune and early_stopping_counter > early_stopping_iter:
            break
    return best_loss, best_matrix


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> ArrayLike:
    min_lr, max_lr = itemgetter('min_lr', 'max_lr')(cfg.hpo)
    params = {
        "initial_lr": trial.suggest_loguniform("initial_lr", min_lr, max_lr)
    }

    accuracies = []
    for fold in fold_loader(cfg=cfg):
        _, conf_matrix = run_training(cfg=cfg, fold=fold, params=params, trial=trial)
        TP, TN, FP, FN = itemgetter('TP', 'TN', 'FP', 'FN')(conf_matrix)
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        accuracies.append(accuracy)
    return np.mean(accuracies)


def run_study(cfg: DictConfig) -> None:
    study = optuna.create_study(study_name=cfg.study_name, storage=cfg.hpo.rdb, direction='maximize', load_if_exists=True)
    ntrials, logger_name = itemgetter('ntrials', 'logger')(cfg.hpo)
    study.optimize(partial(objective, cfg=cfg), n_trials=ntrials)

    best_trial = study.best_trial

    logger = logging.getLogger(logger_name)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")


def view_study(cfg: DictConfig) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    best_trial = study.best_trial
    print(best_trial)

    fig = optuna.visualization.plot_parallel_coordinate(study, params=["activation", "nlayers", "initial_lr"], target_name="Accuracy")
    fig.write_html(cfg.hpo.plot)


def train_w_best_params(cfg: DictConfig) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    best_trial = study.best_trial

    table = load_everything(cfg)
    num_rows = table.num_rows
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    train_len, valid_len = num_rows * train_pct // 100, num_rows * valid_pct // 100
    tables = table.slice(0, train_len), table.slice(train_len, valid_len)

    valid_loss, conf_matrix = run_training(cfg=cfg, fold=tables, params=best_trial.params, save_model=True, prune=False)
    logger = logging.getLogger(cfg.final.logger)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")
    TP, TN, FP, FN = itemgetter('TP', 'TN', 'FP', 'FN')(conf_matrix)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    logger.info(f"Validation loss: {valid_loss} Accuracy: {accuracy}")


def test(cfg: DictConfig) -> None:
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    test_pct = 100 - train_pct - valid_pct
    table = load_everything(cfg)
    num_rows = table.num_rows
    test_offset = num_rows * (100 - test_pct) // 100
    test_table = table.slice(test_offset)
    test_loader = get_dataloaders(test_table)

    model = load_jit_model(cfg)
    engine = Engine(model=model)
    conf_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    engine.evaluate(test_loader, conf_matrix)
    TP, TN, FP, FN = itemgetter('TP', 'TN', 'FP', 'FN')(conf_matrix)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    logger = logging.getLogger(cfg.final.logger)
    logger.info(f"Test Accuracy: {accuracy}")

from folds import fold_loader
import hydra
from omegaconf import DictConfig
from operator import itemgetter
import numpy as np
import torch
from torch import optim
from .model import Model
from .engine import Engine
from .dataset import get_dataloaders
import logging
import optuna
from numpy.typing import ArrayLike
import pyarrow as pa
from typing import Dict, Optional
from optuna.trial import Trial
from functools import partial


def run_training(cfg: DictConfig, fold: pa.Table, params: Optional[Dict], trial: Optional[Trial], save_model: bool = False) -> ArrayLike:
    epochs, device, logger_name = itemgetter("epochs", "device", "logger")(cfg.training)
    model = Model(nfeatures=..., ntargets=..., params=params, trial=trial)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    engine = Engine(model, optimizer, device=device)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    train_path, valid_path = itemgetter("train", "valid")(cfg.datasets)
    train_loader, valid_loader = get_dataloaders(train_path, valid_path)
    logger = logging.getLogger(logger_name)
    for epoch in range(epochs):
        train_loss = engine.train(train_loader)
        valid_loss = engine.evaluate(valid_loader)
        logger.info(f"Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"model.pth")
        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            break
    return best_loss


def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> ArrayLike:
    min_lr, max_lr = itemgetter('min_lr', 'max_lr')(cfg.hpo)
    params = {
        "lr": trial.suggest_loguniform("lr", min_lr, max_lr)
    }
    all_losses = []
    for fold in fold_loader(cfg=cfg):
        tmp_loss = run_training(cfg=cfg, fold=fold, params=params, trial=trial)
        all_losses.append(tmp_loss)
    return np.mean(all_losses)


def run_study(cfg: DictConfig) -> None:
    study = optuna.create_study(direction="minimize")
    ntrials, logger_name = itemgetter('ntrials', 'logger')(cfg.hpo)
    study.optimize(partial(objective, cfg=cfg), n_trials=ntrials)

    best_trial = study.best_trial

    logger = logging.getLogger(logger_name)
    logger.info(f"Best trial values: {best_trial.values}")
    logger.info(f"Best trial params: {best_trial.params}")

    scores = []
    for fold in fold_loader(cfg):
        scr = run_training(cfg=cfg, fold=fold, params=best_trial.params, save_model=True)
        scores.append(scr)
    logger.info(f"Score:{np.mean(scores)}")
    
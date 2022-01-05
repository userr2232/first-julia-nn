import hydra
from omegaconf import DictConfig
from operator import itemgetter
import numpy as np
import torch
from torch import optim
from .model import Model
from .engine import Engine
from .dataset import get_dataloaders
from tqdm import tqdm
import logging
import optuna
from numpy.typing import ArrayLike

@hydra.main(config_path="conf", config_name="config")
def run_training(cfg: DictConfig, fold: int, params: dict, trial, save_model: bool = False) -> ArrayLike:
    epochs, device = itemgetter("epochs", "device")(cfg)
    model = Model(nfeatures=..., ntargets=..., trial=trial, params=params)
    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    engine = Engine(model, optimizer, device=device)

    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0

    train_path, valid_path = itemgetter("train", "valid")(cfg.datasets)
    train_loader, valid_loader = get_dataloaders(train_path, valid_path)

    logger = logging.getLogger("trainer")

    for epoch in tqdm(range(epochs)):
        train_loss = engine.train(train_loader)
        valid_loss = engine.evaluate(valid_loader)
        logger.info(f"Fold: {fold}, Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {valid_loss}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), f"model_{fold}.pth")
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter > early_stopping_iter:
            break
    
    return best_loss

def objective(trial: optuna.trial.Trial) -> ArrayLike:
    params = {
        "lr": trial.suggest_loguniform("lr", 1e-6, 1e-1)
    }
    all_losses = []
    for f_ in range(5):
        tmp_loss = run_training(f_, params)
        all_losses.append(tmp_loss)

    return np.mean(all_losses)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5)

    best_trial = study.best_trial

    logger = logging.getLogger("optuna")
    logger.info(f"Best trial:")
    logger.info(f"\tvalues: {best_trial.values}")
    logger.info(f"\tparams: {best_trial.params}")

    scores = 0
    for j in range(5):
        scr = run_training(j, best_trial.params, )
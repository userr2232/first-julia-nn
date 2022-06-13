from __future__ import annotations
from enum import Enum
from typing import Any, Dict, Optional, List, Union
from omegaconf import DictConfig
from operator import itemgetter
from pathlib import Path
import optuna
import torch
from torch.jit import ScriptModule
import torch.nn as nn
import re
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import joblib


class Activation(Enum):
    ELU = nn.ELU()
    LeakyReLU = nn.LeakyReLU() 
    PReLU = nn.PReLU()
    ReLU = nn.ReLU()
    RReLU = nn.RReLU()
    SELU = nn.SELU()
    CELU = nn.CELU()

    @classmethod
    def builder(cls: Activation, name: str) -> nn.Module:
        return cls.__members__[name].value

    @classmethod
    def dir(cls: Activation) -> List[str]:
        return ['ELU', 'LeakyReLU', 'ReLU', 'RReLU', 'SELU', 'CELU']

class Model(nn.Module):
    def __init__(self, cfg: DictConfig, params: Optional[Dict], trial: Optional[optuna.trial.Trial] = None) -> None:
        super().__init__()
        features, ntargets = itemgetter("features", "ntargets")(cfg.model)
        nfeatures = len(features)
        self.trial = trial
        self.params = params
        self.cfg = cfg
        activation = self.param_getter("activation")
        in_features = nfeatures
        nlayers = self.param_getter("nlayers")
        layers = []
        for i in range(nlayers):
            out_features = self.param_getter(f"n_units_l{i}")
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
            layers.append(activation)
            p = self.param_getter(f"dropout_l{i}")
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(in_features, ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def param_getter(self, param_name: str) -> Any:
        min_nlayers, max_nlayers = itemgetter('min_nlayers', 'max_nlayers')(self.cfg.hpo)
        min_nunits, max_nunits = itemgetter('min_nunits', 'max_nunits')(self.cfg.hpo)
        min_dropout, max_dropout = itemgetter('min_dropout', 'max_dropout')(self.cfg.hpo)
        def trial_suggest(param_name: str) -> Any:
            if param_name == "activation":
                return Activation.builder(self.trial.suggest_categorical(param_name, Activation.dir()))
            elif param_name == "nlayers":
                return self.trial.suggest_int(param_name, min_nlayers, max_nlayers)
            elif re.match(r"^n_units_l\d+$", param_name):
                return self.trial.suggest_int(param_name, min_nunits, max_nunits)
            elif re.match(r"^dropout_l\d+$", param_name):
                return self.trial.suggest_float(param_name, min_dropout, max_dropout)
            else:
                raise ValueError("Invalid parameter name.")
        try:
            return self.params[param_name] if param_name != "activation" else Activation.builder(self.params[param_name])
        except KeyError:
            return trial_suggest(param_name)


def save_jit_model(cfg: DictConfig, model: nn.Module) -> None:
    path = Path(cfg.model.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.cpu()
    sample_input_cpu = torch.rand(len(cfg.model.features))
    traced_cpu = torch.jit.trace(cpu_model, sample_input_cpu)
    torch.jit.save(traced_cpu, Path(cfg.model.path) / cfg.model.nn_checkpoint)


def load_jit_model(cfg: DictConfig) -> ScriptModule:
    path = Path(cfg.model.path)
    return torch.jit.load(path / cfg.model.nn_checkpoint)


def Scaler(df: pd.DataFrame, columns: List[str], save: Optional[bool] = False, path: Optional[Union[str,Path]] = None) -> MinMaxScaler:
    if save:
        if path is None:
            raise ValueError("argument path cannot be None if save is True.")
    scaler = MinMaxScaler((0,1))
    if save:
        joblib.dump(scaler, path)
    return scaler.fit(df.loc[:, columns])
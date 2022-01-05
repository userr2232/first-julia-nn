from __future__ import annotations
from enum import Enum
from typing import Any, List

import optuna
import torch.nn as nn
import re


class Model(nn.Module):
    class Activation(Enum):
        ELU = nn.ELU()
        LeakyReLU = nn.LeakyReLU() 
        PReLU = nn.PReLU()
        ReLU = nn.ReLU()
        RReLU = nn.RReLU()
        SELU = nn.SELU()
        CELU = nn.CELU()

        @classmethod
        def builder(cls: Model.Activation, name: str) -> nn.Module:
            return cls.__members__[name].value

    def param_getter(self, param_name: str) -> Any:
        def trial_suggest(param_name: str) -> Any:
            if param_name == "activation":
                return Model.Activation.builder(self.trial.suggest_categorical(param_name, dir(Model.Activation)))
            elif param_name == "nlayers":
                return self.trial.suggest_int(param_name, 1, 20)
            elif re.match(r"^n_units_l\d+$", param_name):
                return self.trial.suggest_int(param_name, 1<<2, 1<<7)
            elif re.match(r"^dropout_l\d+$", param_name):
                return self.trial.suggest_float(param_name, 0.05, 0.5)
            else:
                raise ValueError("Invalid parameter name.")
        return self.params.get(param_name, trial_suggest(param_name))

    def __init__(self, nfeatures: int, ntargets: int, trial: optuna.trial.Trial, params: dict) -> None:
        super().__init__()
        self.trial = trial
        self.params = params
        activation = self.param_getter("activation")
        in_features = nfeatures
        nlayers = self.param_getter("nlayers")
        layers = []
        for i in range(nlayers):
            out_features = self.param_getter(f"n_units_l{i}")
            layers.append(nn.Linear(in_features, out_features))
            in_features = out_features
            layers.append(activation)
            p = trial.suggest_float(f"dropout_l{i}", 0.05, 0.5)
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(in_features, ntargets))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
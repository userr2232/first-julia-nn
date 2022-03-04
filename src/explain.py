import shap
import pandas as pd
from src.folds import load_everything
from omegaconf import DictConfig
from pathlib import Path
from src.inference import predict
from src.dataset import processing
from functools import partial
import torch
import numpy as np


def F(X, cfg: DictConfig) -> None:
    X = torch.from_numpy(X).to(torch.float32)
    return predict(cfg, inputs=X).flatten().detach().numpy()


def explain(cfg: DictConfig) -> None:
    df = processing(load_everything(cfg).to_pandas()).loc[:, cfg.model.features].copy()
    X = df.sample(n=cfg.explanation.sample_size, random_state=42)
    explainer = shap.KernelExplainer(partial(F, cfg=cfg), X)
    shap_values = explainer.shap_values(X, nsamples=500)
    shap.summary_plot(shap_values, X)

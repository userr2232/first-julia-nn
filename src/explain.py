import shap
import pandas as pd
from src.folds import load_everything
from omegaconf import DictConfig
from pathlib import Path
from src.inference import predict
from src.dataset import processing
from functools import partial
import torch
import matplotlib.pyplot as plt


def F(X, cfg: DictConfig) -> None:
    X = torch.from_numpy(X).to(torch.float32)
    return predict(cfg, inputs=X).flatten().detach().numpy()


def explain(cfg: DictConfig) -> None:
    df = processing(columns=cfg.model.features, 
                        df=load_everything(cfg).to_pandas())[0].loc[:, cfg.model.features].copy()
    X = df.sample(n=cfg.explanation.sample_size, random_state=42)

    explainer = shap.KernelExplainer(partial(F, cfg=cfg), X)
    shap_values = explainer.shap_values(X, nsamples=1000)
    plt.subplots_adjust(left=0.2, bottom=0.12, right=0.97, top=0.95, wspace=0.04, hspace=0.1)
    shap.summary_plot(shap_values, X,
                        feature_names=["foF2", "h'F", "prev. h'F", 
                                        "∆h'F/∆t", "F10.7", "F10.7 (90 d)",
                                        "ap", "ap (24 h)", "DNS", "DNC"])

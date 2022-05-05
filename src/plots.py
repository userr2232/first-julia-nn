import optuna
from omegaconf import DictConfig
import pandas as pd
from typing import List
from matplotlib import pyplot as plt
import plotly.express as px


def optimization_history(cfg: DictConfig, fontsize: int) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    fig = optuna.visualization.plot_optimization_history(study, target_name="avg. accuracy")
    fig.update_layout(font=dict(size=fontsize))
    fig.show()

def plot_parallel_coordinate(cfg: DictConfig, fontsize: int, m: int) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    fig = optuna.visualization.plot_parallel_coordinate(study, params=["activation", "initial_lr", "nlayers"], target_name="avg. accuracy")
    fig.update_layout(font=dict(size=fontsize), margin=dict(l=m, r=m, t=m, b=m))
    fig.show()


def optuna_plots(cfg: DictConfig) -> None:
    plot_parallel_coordinate(cfg, 30, 120)
    optimization_history(cfg, 40)


def axvspan_early_ESF(df: pd.DataFrame, col: str, levels: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(30, 5))
    plt.plot(df['LT_x'], df[col], alpha=0)
    ax.get_yaxis().set_visible(False)
    for level in levels:
        for (x, _), g in df.groupby([col, df[col].ne(level).cumsum()]):
            if x == level:
                start = g.date.min()
                end = g.date.max()
                plt.axvspan(start, end, color='r', linewidth=0)


def plotly_axvspan_early_ESF(df: pd.DataFrame, col: str, levels: List[int]) -> None:
    fig = px.line(df)
    for level in levels:
        for x, g in df.groupby([col, df[col].ne(level).cumsum()]):
            if x[col] == level:
                start = g.index.min()
                end = g.index.max()
                fig.add_vrect(x0=start, x1=end,fillcolor="green", opacity=0.25, line_width=0)

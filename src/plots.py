import optuna
from omegaconf import DictConfig


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
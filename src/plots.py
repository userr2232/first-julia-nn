import optuna
from omegaconf import DictConfig
import pandas as pd
from typing import List
import matplotlib as mpl
from matplotlib import pyplot as plt
import plotly.express as px
from src.folds import load_everything
from src.dataset import get_test_dataloader
from operator import itemgetter
from pathlib import Path
from src.model import load_jit_model
from src.engine import Engine
import numpy as np
from matplotlib.axes import Axes
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches


plt.rcParams.update({"font.family": "serif", "font.serif": ["Palatino"]})

def optimization_history(cfg: DictConfig, fontsize: int) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    fig = optuna.visualization.plot_optimization_history(study, target_name="avg. accuracy")
    fig.update_layout(font=dict(size=fontsize,family="Palatino"))
    fig.show()


def plot_parallel_coordinate(cfg: DictConfig, fontsize: int, m: int) -> None:
    study = optuna.load_study(study_name=cfg.study_name, storage=cfg.hpo.rdb)
    fig = optuna.visualization.plot_parallel_coordinate(study, params=["activation", "initial_lr", "nlayers"], target_name="avg. accuracy")
    fig.update_layout(font=dict(size=fontsize, family="Palatino"), margin=dict(l=m, r=m, t=m, b=m))
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


def confusion_calendar(df: pd.DataFrame) -> None:
    def prediction_mapper(row):
        if np.isnan(row.TP): return np.NaN
        if row.TP: return 4
        if row.FP: return 3
        if row.TN: return 2
        if row.FN: return 1
        return -1

    def calmapv3(ax: Axes, df: pd.DataFrame):
        bins = 4
        cmap = plt.cm.spring
        cmaplist = [cmap(i) for i in np.linspace(0, 255, bins).astype(np.int32)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list('helado tricolor', cmaplist, bins)
        bounds = np.linspace(0, bins, bins+1) + 0.5

        # data must be a dataframe with dates as index. It must contain all days of the year
        year = df.index[0].year

        ax.tick_params('x', length=0, labelsize="medium", which='major')
        ax.tick_params('y', length=0, labelsize="x-small", which='major')

        # Month borders
        xticks, labels = [], []
        start = datetime(year,1,1).weekday()

        data = np.ones(7*53)
        data[start:start+len(df)] = df.iloc[:].values.ravel()
        data = data.reshape(53, 7).T

        for month in range(1,13):
            first = datetime(year, month, 1)
            last = first + relativedelta(months=1, days=-1)

            y0 = first.weekday()
            y1 = last.weekday()

            x0 = (first.timetuple().tm_yday + start - 1) // 7
            x1 = (last.timetuple().tm_yday + start - 1) // 7

            P = [ (x0,   y0), (x0,    7),  (x1,   7),
                (x1,   y1+1), (x1+1,  y1+1), (x1+1, 0),
                (x0+1,  0), (x0+1,  y0) ]
            xticks.append(x0 +(x1-x0+1)/2)
            labels.append(first.strftime("%b"))
            poly = Polygon(P, edgecolor="black", facecolor="None",
                        linewidth=1, zorder=20, clip_on=False)
            ax.add_artist(poly)
        
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)
        ax.set_yticks(0.5 + np.arange(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_title("{}".format(year), weight="semibold")
        
        # Clearing first and last day from the data
        valid = datetime(year, 1, 1).weekday()
        data[:valid,0] = np.nan
        valid = datetime(year, 12, 31).weekday()
        # data[:,x1+1:] = np.nan
        data[valid+1:,x1] = np.nan

        # Showing data
        ax.imshow(data, extent=[0,53,0,7], zorder=10, vmin=1, vmax=4,
                cmap=cmap, origin="lower", alpha=.75)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    date_range = pd.Series(pd.date_range(pd.Timestamp('2019-01-01'), pd.Timestamp('2020-12-31')), name='LT') + pd.Timedelta(19, unit='hours') + pd.Timedelta(30, unit='min')
    df = pd.merge_asof(date_range, df, on='LT', tolerance=pd.Timedelta('1min'))
    df['prediction'] = df.apply(prediction_mapper, axis=1)
    df.drop(['TP', 'FP', 'TN', 'FN'], inplace=True, axis=1)
    _, ax = plt.subplots(2, 1, figsize=(10, 4), tight_layout=True)
    df.index = df.LT.dt.date
    calmapv3(ax=ax[0], df=df.loc[(df.LT.dt.year == 2019)].drop('LT', axis=1))
    calmapv3(ax=ax[1], df=df.loc[(df.LT.dt.year == 2020)].drop('LT', axis=1))
    plt.show()


def nn_confusion_calendar(cfg: DictConfig) -> None:
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    test_pct = 100 - train_pct - valid_pct
    table = load_everything(cfg)
    num_rows = table.num_rows
    test_offset = num_rows * (100 - test_pct) // 100
    test_table = table.slice(test_offset)
    test_df = test_table.to_pandas()
    test_df['LT'] = test_df['LT-1h']
    test_loader = get_test_dataloader(cfg.model.features, test_df, scaler_checkpoint=Path(cfg.model.path) / cfg.model.scaler_checkpoint)

    model = load_jit_model(cfg)
    engine = Engine(model=model)
    test_df = engine.evaluate_with_LT(test_loader)
    print(test_df)
    print("TP:", test_df.TP.sum())
    print("FP:", test_df.FP.sum())
    print("TN:", test_df.TN.sum())
    print("FN:", test_df.FN.sum())
    confusion_calendar(test_df)


def first_confusion_calendar(cfg: DictConfig) -> None:
    train_pct, valid_pct = itemgetter('train', 'valid')(cfg.final.split)
    test_pct = 100 - train_pct - valid_pct
    table = load_everything(cfg)
    num_rows = table.num_rows
    test_offset = num_rows * (100 - test_pct) // 100
    test_table = table.slice(test_offset)
    test_df = test_table.to_pandas()
    test_df.rename(columns={'LT_y': 'LT'}, inplace=True)
    test_df.loc[:, 'hF_thr'] = test_df.apply(lambda x: x['F10.7']*1.08 + 200.3, axis=1)
    test_df.loc[test_df.loc[(test_df.loc[:, 'V_hF'] > (test_df.loc[:, 'hF_thr'] + 10))].index, 'spreadF_prediction'] = 1
    test_df.loc[test_df.loc[(test_df.loc[:, 'V_hF'] < (test_df.loc[:, 'hF_thr'] - 10))].index, 'spreadF_prediction'] = 0
    test_df.loc[test_df.loc[test_df.spreadF_prediction.notna()].index, 'spreadF_prediction'] = test_df.spreadF_prediction > 0
    test_df.loc[((test_df.spreadF_prediction.notna())&(test_df.accum_ESF.notna())), 'TP'] = ((test_df.spreadF_prediction == True) & (test_df.accum_ESF > 0))
    test_df.loc[((test_df.spreadF_prediction.notna())&(test_df.accum_ESF.notna())), 'FP'] = ((test_df.spreadF_prediction == True) & (test_df.accum_ESF == 0))
    test_df.loc[((test_df.spreadF_prediction.notna())&(test_df.accum_ESF.notna())), 'TN'] = ((test_df.spreadF_prediction == False) & (test_df.accum_ESF == 0))
    test_df.loc[((test_df.spreadF_prediction.notna())&(test_df.accum_ESF.notna())), 'FN'] = ((test_df.spreadF_prediction == False) & (test_df.accum_ESF > 0))
    print(f"could not predict {(test_df.spreadF_prediction.isna()&(test_df.accum_ESF.notna())).sum()} days")
    print("TP:", test_df.TP.sum())
    print("FP:", test_df.FP.sum())
    print("TN:", test_df.TN.sum())
    print("FN:", test_df.FN.sum())
    test_df.drop(test_df.columns.difference(['LT', 'TP', 'FP', 'TN', 'FN']), axis=1, inplace=True)
    print(test_df)
    confusion_calendar(test_df)


def confusion_calendar_patches():
    bins = 4
    cmap = plt.cm.spring
    cmaplist = [cmap(i) for i in np.linspace(0, 255, bins).astype(np.int32)]
    yellow_patch = mpatches.Patch(color=cmaplist[-1], label='True Positive')
    orange_patch = mpatches.Patch(color=cmaplist[-2], label='False Positive')
    pink_patch = mpatches.Patch(color=cmaplist[-3], label='True Negative')
    pinker_patch = mpatches.Patch(color=cmaplist[-4], label='False Negative')
    _, _ = plt.subplots(1, 1, figsize=(3,2))
    plt.legend(handles=[yellow_patch, orange_patch, pink_patch, pinker_patch], 
                bbox_to_anchor=(0.5, 0.5), loc='center', borderaxespad=0.)
    plt.tight_layout()
    plt.axis('off')
    plt.show()

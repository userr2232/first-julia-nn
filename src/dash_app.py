from omegaconf import DictConfig, OmegaConf
from typing import Tuple, List, Final
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html
import datetime
from numpy.typing import ArrayLike
from dateutil.relativedelta import relativedelta
import plotly.express as px
from pathlib import Path
import numpy as np
from plotly.subplots import make_subplots
import dash
from dash.dependencies import Input, Output
from operator import itemgetter


def display_year(df: pd.DataFrame, year: int, fig: go.Figure, row: int) -> None:
    start = datetime.date(year, 1, 1).weekday()
    Z = np.zeros(7 * 53) * np.nan
    Z[start:start+len(df)] = df.val
    Z = Z.reshape(53, 7).T

    colorscale = [(0, 'white'), (1e-12, 'white'),
                    (1e-12, '#ff00ff'), (0.15, '#ff00ff'), 
                    (0.15, "#ff55aa"), (0.3, '#ff55aa'),
                    (0.3, '#ffaa55'), (0.45, '#ffaa55'),
                    (0.45, '#ffff00'), (0.6, '#ffff00'),
                    (0.6, '#03dddc'), (0.75, '#03dddc'),
                    (0.75, '#000000'), (1, '#000000')]

    def formatter(row):
        return ('' if np.isnan(row.val) else '<br>'.join([
            str(row.date.date()),
            f"h\'F: {row.V_hF:.2f}",
            f"h\'F (prev. 30 min): {row.V_hF_prev:.2f}",
            f"F10.7: {row['F10.7']:.2f}",
            f"F10.7 (avg. 90 days): {row['F10.7 (90d)']:.2f}",
            f"ap: {row.AP:.0f}",
            f"ap (24h): {row['AP (24h)']:.2f}",
        ])) + '<extra></extra>'
    
    customdata = [''] * 7 * 53
    customdata[start:start+len(df)] = df.apply(formatter, axis=1).tolist()
    
    customdata = np.array(customdata).reshape(53, 7).T

    plots = [go.Heatmap(
        z=Z,
        zmin=0, zmax=1,
        colorscale=colorscale,
        xgap=1, ygap=1,
        showscale=False,
        customdata=customdata,
        hovertemplate='%{customdata}'
    )]

    xticks, xlabels = [], []
    
    kwargs = {
        "mode": "lines",
        "hoverinfo": "skip",
        "line": {"color": "black", "width": 2}
    }
    for month in range(1, 13):
        first = datetime.date(year, month, 1)
        last = first + relativedelta(months=1, days=-1)

        x0, x1 = (first.timetuple().tm_yday + start - 1) // 7, (last.timetuple().tm_yday + start - 1) // 7
        y0, y1 =  first.weekday(), last.weekday()

        x = np.array([x0, x0, x1, x1, x1+1, x1+1, x0+1, x0+1, x0]) - 0.5
        y = np.array([y0, 7, 7, y1+1, y1+1, 0, 0, y0, y0]) - 0.5
        
        xticks.append(x0 - 0.5 + (x1 - x0 + 1) / 2)
        xlabels.append(first.strftime("%b"))
        plots.append(go.Scatter(x=x, y=y, **kwargs))
    
    layout = go.Layout(
        title="Spread F predictions",
        yaxis=dict(
            showline=False, showgrid=False,
            zeroline=False, tickmode='array',
            ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            tickvals=[0, 1, 2, 3, 4, 5, 6],
            autorange="reversed"
        ),
        xaxis=dict(
            showline=False, showgrid=False,
            zeroline=False, tickmode='array',
            ticktext=xlabels, tickvals=xticks,
        ),
        font={'size': 20, 'color': 'black'},
        plot_bgcolor=('#fff'),
        showlegend=False
    )

    fig.add_traces(plots, rows=[(row+1)]*len(plots), cols=[1]*len(plots))
    fig.update_layout(layout)
    fig.update_layout(title_font_size=40)
    fig.update_xaxes(layout['xaxis'], tickfont_size=30)
    fig.update_yaxes(layout['yaxis'])


def display_years(DFs: List[pd.DataFrame], years: List[int]) -> go.Figure:
    N: Final[int] = len(years)
    day_counter = 0
    fig = make_subplots(rows=N, cols=1, subplot_titles=years)
    for i, (df, year) in enumerate(zip(DFs, years)):
        display_year(df, year=year, fig=fig, row=i)
    fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text}</b>"))
    fig.update_annotations(font_size=28)
    fig.update_layout(height=350*N)
    return fig


def read_calendar_dataset() -> Tuple[List[ArrayLike], List[int]]:
    def prediction_mapper(row):
        if np.isnan(row.TP) and np.isnan(row.prediction): return np.nan
        if not np.isnan(row.TP):
            if row.TP: return 4
            if row.FP: return 3
            if row.TN: return 2
            if row.FN: return 1
        else:
            if row.prediction: return 5
            else: return 6

    confusion_path, nn_inputs_path, predictions_path = Path(cfg.datasets.confusion), Path(cfg.datasets.nn_inputs), Path(cfg.datasets.predictions)
    confusion_df = pd.read_csv(confusion_path, parse_dates=['date'], infer_datetime_format=True)
    inputs_df = pd.read_csv(nn_inputs_path)
    predictions_df = pd.read_csv(predictions_path)
    df = pd.merge(inputs_df, predictions_df, on='day_idx', how='left')
    df = pd.merge(df, confusion_df, on='day_idx', how='left')
    df.date = pd.Timestamp(year=1970, month=1, day=1) + pd.to_timedelta(df.day_idx, unit='days')
    years = np.unique(df.date.dt.year).tolist()
    DFs = []
    for year in years:
        start, end = datetime.date(year, 1, 1), datetime.date(year, 12, 31)
        dt_range = pd.date_range(start, end)
        arr = np.empty(len(dt_range))
        dt_range_df = pd.DataFrame(arr)
        dt_range_df['date'] = dt_range.copy()
        dt_df = pd.merge(dt_range_df, df, on='date', how='left')
        dt_df['val'] = dt_df.apply(prediction_mapper, axis=1) / 7
        DFs.append(dt_df.drop(0, axis=1))
    return DFs[::-1], years[::-1]


def dash_app():
    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(id='confusion-calendar', config={'displayModeBar': False}),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
        )
    ])
    return app

if __name__ == "__main__":
    _cfg = OmegaConf.load('conf/config.yaml')
    _cfg.root = str(Path.cwd())
    cfg = _cfg
    app = dash_app()
    @app.callback(Output('confusion-calendar', 'figure'), Input('interval-component', 'n_intervals'))
    def update_calendar(n):
        DFs, years = read_calendar_dataset()
        fig = display_years(DFs, years)
        return fig
    app.run_server(debug=True)
from hydra import compose, initialize
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from operator import itemgetter
import h5py
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.dataset as ds
from pyarrow import fs
from typing import Dict, Union, Optional
from itertools import product
import os
import re


def preprocessing(cfg: DictConfig, save: bool = False, path: Optional[Union[str, Path]] = None) -> pa.Table:
    if save:
        assert(path is not None)
        path = Path(path)
    
    JULIA_PATH = Path(cfg.datasets.julia)

    pattern = r"^\d\d_\d\d\d\d.csv$"
    re_obj = re.compile(pattern)
    _, _, filenames = next(os.walk(JULIA_PATH), (None, None, []))
    fabiano_ESF = pd.DataFrame({})
    for filename in filenames:
        if re_obj.fullmatch(filename):
            season_df = pd.read_csv(JULIA_PATH / filename, parse_dates=['LT'], infer_datetime_format=True)
            if not season_df.empty:
                fabiano_ESF = pd.concat([fabiano_ESF, season_df], ignore_index=True)
    fabiano_ESF.LT = pd.to_datetime(fabiano_ESF.LT)

    fabiano_ESF.sort_values('LT', inplace=True)
    fabiano_ESF.reset_index(drop=True, inplace=True)
    h5_d = h5py.File(cfg.datasets.sao, 'r')
    df = pd.DataFrame(h5_d['Data']['GEO_param'][()])
    df.loc[:, 'date_hour'] = pd.to_datetime(df.loc[:, ('YEAR', 'MONTH', 'DAY', 'HOUR')])
    df['LT'] = df.date_hour - pd.Timedelta("5h")
    df.drop('date_hour', axis=1, inplace=True)
    df.sort_values('LT', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(['YEAR', 'MONTH', 'DAY', 'HOUR'], axis=1, inplace=True)
    df2 = pd.DataFrame(h5_d['Data']['SAO_total'][()])
    df2.rename(columns={'MIN': 'MINUTE', 'SEC': 'SECOND'}, inplace=True)
    df2.loc[:, 'datetime'] = pd.to_datetime(df2.loc[:,('YEAR','MONTH','DAY','HOUR','MINUTE')])
    df2.sort_values('datetime', inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df2['LT'] = df2.datetime - pd.Timedelta('5h')
    df2.drop('datetime', axis=1, inplace=True)
    df2.drop(['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'foF1',
    'hmF1','foE','hmE','V_hF2','V_hE','V_hEs','hmF2'], axis=1, inplace=True)
    df2.dropna(inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df.drop(['DST', 'KP'], axis=1, inplace=True)
    df2.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df3 = pd.merge_asof(df2, df, on='LT', tolerance=pd.Timedelta('59m'))
    df3.index = df3.LT
    df3.drop(df3.loc[((df3['F10.7'].isna())|(df3.AP.isna()))].index, inplace=True)
    df3['F10.7 (90d)'] = df3['F10.7'].rolling('90d').mean()
    df3['F10.7 (90d dev.)'] = df3['F10.7'] - df3['F10.7 (90d)']
    df3['AP (24h)'] = df3['AP'].rolling('24h').mean()
    df3['V_hF_prev'] = df3['V_hF'].rolling('30min').agg(lambda rows: rows[0])
    df3['V_hF_prev_time'] = df3['V_hF'].rolling('30min').agg(lambda rows: pd.to_datetime(rows.index[0]).value)
    df3['V_hF_prev_time'] = pd.to_datetime(df3['V_hF_prev_time'])
    df3.reset_index(drop=True, inplace=True)
    df3['delta_hF'] = df3['V_hF']-df3['V_hF_prev']
    df3['delta_time'] = (df3['LT']-df3['V_hF_prev_time']).dt.components.minutes + 1e-9
    df3['delta_hF_div_delta_time'] = df3['delta_hF'] / df3['delta_time']
    df3.drop(['delta_hF', 'delta_time', 'V_hF_prev_time'], axis=1, inplace=True)
    delta_hours = cfg.preprocessing.delta_hours
    fabiano_ESF[f'LT-{delta_hours}h'] = fabiano_ESF.LT - pd.Timedelta(f'{delta_hours}h')
    fabiano_ESF.index = fabiano_ESF.LT
    fabiano_ESF['accum_ESF'] = fabiano_ESF.ESF.rolling('1h').sum()
    fabiano_ESF.reset_index(drop=True, inplace=True)
    merged = pd.merge_asof(fabiano_ESF, df3, 
                            left_on=f'LT-{delta_hours}h', right_on='LT', 
                            tolerance=pd.Timedelta('15m'), 
                            direction='nearest')
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    FIRST2_0 = merged.loc[((merged.LT_y.dt.hour == 19)&(merged.LT_y.dt.minute == 30))].copy()
    FIRST2_0.reset_index(drop=True, inplace=True)
    FIRST2_0['year'] = FIRST2_0.LT_y.dt.year
    print(FIRST2_0)
    data = pa.Table.from_pandas(FIRST2_0)
    if save:
        partitioning = ds.partitioning(pa.schema([("year", pa.int16())]), flavor="hive")
        ds.write_dataset(data, str(path / "partitioned"), format="ipc", 
                         partitioning=partitioning, existing_data_behavior="delete_matching")
    return data


def create_partitions(cfg: DictConfig):
    processed_dir = cfg.datasets.processed
    return preprocessing(cfg, save=True, path=Path(processed_dir))

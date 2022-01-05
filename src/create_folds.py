from hydra import compose, initialize
from pathlib import Path
import pandas as pd
import seaborn as sns
import numpy as np
from operator import itemgetter
from utils import date_parser
import h5py
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.dataset as ds
from typing import Union, Optional

def preprocessing(cfg: DictConfig, save: bool = False, path: Optional[Union[str, Path]] = None) -> pa.Table:
    if save: assert(path is not None)
    path = Path(path)
    fabiano_ESF = pd.DataFrame(columns=['LT', 'ESF'])
    months = ['03', '06', '09', '12']
    START_YEAR, END_YEAR = itemgetter('start', 'end')(cfg.years)
    years = range(START_YEAR, END_YEAR)
    JULIA_PATH = Path(cfg.datasets.julia)
    for month in months:
        for year in years:
            df = pd.read_csv(JULIA_PATH / f"{month}_{year}.csv", parse_dates=['LT'], date_parser=date_parser)
            fabiano_ESF = pd.concat([fabiano_ESF, df])
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
    df2.drop(['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'], axis=1, inplace=True)
    df3 = pd.merge_asof(df2, df, on='LT', tolerance=pd.Timedelta('59m'))
    df3.index = df3.LT
    df3.drop(df3.loc[((df3['F10.7'].isna())|(df3.AP.isna()))].index, inplace=True)
    df3['F10.7 (90d)'] = df3['F10.7'].rolling('90d').mean()
    df3['AP (24h)'] = df3['AP'].rolling('24h').mean()
    df3['V_hF_prev'] = df3['V_hF'].rolling('30min').agg(lambda rows: rows[0])
    df3.reset_index(drop=True, inplace=True)
    df3.drop(columns=['foF1','foF2','hmF1','foE','hmE','V_hF2','V_hE','V_hEs','DST'], inplace=True)
    fabiano_ESF['LT-1h'] = fabiano_ESF.LT - pd.Timedelta('1h')
    fabiano_ESF.index = fabiano_ESF.LT
    fabiano_ESF['accum_ESF'] = fabiano_ESF.rolling('1h').sum()
    fabiano_ESF.reset_index(drop=True, inplace=True)
    merged = pd.merge_asof(fabiano_ESF, df3, 
                            left_on='LT-1h', right_on='LT', 
                            tolerance=pd.Timedelta('15m'), 
                            direction='nearest')
    merged.dropna(inplace=True)
    merged.reset_index(drop=True, inplace=True)
    FIRST2_0 = merged.loc[((merged.LT_y.dt.hour == 19)&(merged.LT_y.dt.minute == 30))].copy()
    FIRST2_0.reset_index(drop=True, inplace=True)
    FIRST2_0['year'] = FIRST2_0.LT_y.dt.year
    data = pa.Table.from_pandas(FIRST2_0)
    if save:
        ds.write_dataset(data, str(path / "partitioned"), format="ipc",
                            partitioning=ds.partitioning(pa.schema([("year", pa.int16())])))
    return data


if __name__ == "__main__":
    with initialize(config_path="../conf", job_name="fold_creation"):
        cfg = compose(config_name="config")
        FIRST2_0 = preprocessing(cfg, save=True, path=Path(cfg.datasets.processed_dir))
        

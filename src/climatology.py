import pandas as pd
from typing import Union
from pathlib import Path
from src.types import Month
import re
import os


"""
    This function returns the days for which ESF occurred before 19:30 LT (Local Time)
    The csv files have the pattern: MM_yyyy.csv
    path: path to the folder containing the csv files for each season
    return: a dataframe with the days for which ESF occurred before 19:30 LT and after 7:00 LT 
"""
def days_of_early_ESF(path: Union[str,Path]) -> pd.DataFrame:
    path = Path(path)
    pattern = r"^\d\d_\d\d\d\d.csv$"
    re_obj = re.compile(pattern)
    _, _, filenames = next(os.walk(path), (None, None, []))
    df = pd.DataFrame({})
    for filename in filenames:
        if re_obj.fullmatch(filename):
            season_df = pd.read_csv(path / filename, parse_dates=['LT'], infer_datetime_format=True)
            if not season_df.empty:
                season_df = season_df.loc[((season_df.LT.dt.hour > 7)&
                                                ((season_df.LT.dt.hour == 19) & (season_df.LT.dt.minute < 30)))].copy()
                df = pd.concat([df, season_df], ignore_index=True)
    return df.loc[(df.ESF > 0)].sort_values('LT').reset_index(drop=True)

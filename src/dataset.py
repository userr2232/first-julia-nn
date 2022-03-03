import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional
import pyarrow as pa
import numpy as np
import joblib
import pandas as pd


def processing(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()

    columns = ['V_hF', 'V_hF_prev', 'F10.7', 'F10.7 (90d)', 'AP', 'AP (24h)']
    
    scaler_path = "/Users/userr2232/Documents/misc/first-julia-nn/models/scaler.gz"
    scaler = joblib.load(scaler_path)
    new_df[columns] = scaler.transform(new_df.loc[:, columns])

    D = new_df.LT_y.dt.dayofyear
    new_df['DNS'] = np.sin(2*np.pi*D/365)
    new_df['DNC'] = np.cos(2*np.pi*D/365)

    new_df['ESF_binary'] = (new_df['accum_ESF'] > 0)*1

    return new_df.loc[:, new_df.columns.intersection(columns + ['DNS', 'DNC', 'ESF_binary'])].copy()


class JuliaDataset(Dataset):
    def __init__(self, table: pa.Table):
        self.df = processing(table.to_pandas())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        return torch.tensor(row.drop(labels='ESF_binary'), dtype=torch.float32, requires_grad=True), \
                torch.tensor([row.ESF_binary], dtype=torch.float32, requires_grad=True)


def get_dataloaders(*args: pa.Table, **kwargs) -> Union[List[DataLoader], DataLoader]:
    if len(args) == 1:
        return DataLoader(dataset=JuliaDataset(args[0]), **kwargs)
    return [ DataLoader(dataset=JuliaDataset(table), **kwargs) for table in args ]
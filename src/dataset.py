import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, List, Optional, Tuple
import pyarrow as pa
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.model import Scaler


def processing(columns: List[str], df: pd.DataFrame, scaler: Optional[Union[MinMaxScaler,str,Path]] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    new_df = df.copy()

    columns = pd.Index(columns).intersection(new_df.columns, sort=False).tolist()
    if scaler is None:
        scaler = Scaler(df=df, columns=columns)
    else:
        if isinstance(scaler, str) or isinstance(scaler, Path):
            scaler: MinMaxScaler = joblib.load(scaler)
        elif not isinstance(scaler, MinMaxScaler):
            raise ValueError("scaler argument must be of type MinMaxScaler or str or Path")
    
    new_df[columns] = scaler.transform(new_df.loc[:, columns])

    D = new_df.LT_y.dt.dayofyear
    new_df['DNS'] = np.sin(2*np.pi*D/365)
    new_df['DNC'] = np.cos(2*np.pi*D/365)

    new_df['ESF_binary'] = (new_df['accum_ESF'] > 0)*1

    return new_df.loc[:, new_df.columns.intersection(columns + ['DNS', 'DNC', 'ESF_binary'])].copy(), scaler


class JuliaDataset(Dataset):
    def __init__(self, columns: List[str], table: pa.Table, scaler: Optional[Union[MinMaxScaler,str,Path]] = None):
        self.df, self.scaler = processing(columns=columns, df=table.to_pandas(), scaler=scaler)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        return torch.tensor(row.drop(labels='ESF_binary'), dtype=torch.float32, requires_grad=True), \
                torch.tensor([row.ESF_binary], dtype=torch.float32, requires_grad=True)


def get_dataloaders(columns:List[str], *args: pa.Table, scaler: Optional[Union[MinMaxScaler,str,Path]] = None, **kwargs) -> Union[List[DataLoader], DataLoader]:
    if len(args) == 1:
        return DataLoader(dataset=JuliaDataset(columns=columns, table=args[0], scaler=scaler), **kwargs)
    else:
        training_dataset = JuliaDataset(columns=columns, table=args[0], scaler=scaler)
        scaler: MinMaxScaler = training_dataset.scaler
        datasets = [ training_dataset ] + [ JuliaDataset(columns=columns, table=table, scaler=scaler) for table in args[1:] ]
        return [ DataLoader(dataset, **kwargs) for dataset in datasets ]
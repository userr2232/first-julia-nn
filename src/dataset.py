import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union
import pyarrow as pa
import numpy as np


class JuliaDataset(Dataset):
    def __init__(self, table: pa.Table):
        self.df = table.to_pandas()
        self.preprocessing()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return torch.tensor(row[['V_hF', 'V_hF_prev', 'F10.7', 'F10.7 (90d)', 'AP', 'AP (24h)', 'DNS', 'DNC']], dtype=torch.float32, requires_grad=True), \
                torch.tensor([row.ESF_binary], dtype=torch.float32, requires_grad=True)

    def preprocessing(self):
        D = self.df.LT_y.dt.dayofyear
        self.df['DNS'] = np.sin(2*np.pi*D/365)
        self.df['DNC'] = np.cos(2*np.pi*D/365)
        H = self.df.LT_y.dt.hour
        self.df['HS'] = np.sin(2*np.pi*H/24)
        self.df['HC'] = np.cos(2*np.pi*H/24)
        self.df.drop(['LT_x', 'LT-1h', 'LT_y', 'hmF2', 'year'], axis=1, inplace=True)
        self.df['ESF_binary'] = (self.df['accum_ESF'] > 0)*1
        print(self.df)

def get_dataloaders(*args: pa.Table, **kwargs):
    return [ DataLoader(dataset=JuliaDataset(table), **kwargs) for table in args ]
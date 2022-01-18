from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union
import pyarrow as pa


class JuliaDataset(Dataset):
    def __init__(self, table: pa.Table):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass

def get_dataloaders(*args: pa.Table, **kwargs):
    return [ DataLoader(dataset=MyDataset(table), **kwargs) for table in args ]
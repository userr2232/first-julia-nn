from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union

class MyDataset(Dataset):
    def __init__(self, path: Union[str, Path]):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx: int):
        pass

def get_dataloaders(*args: Union[str, Path], **kwargs):
    return [ DataLoader(dataset=MyDataset(path), **kwargs) for path in args ]
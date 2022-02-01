from omegaconf import DictConfig
from operator import itemgetter
import pyarrow as pa
import pyarrow.dataset as ds
from itertools import product
from typing import Iterator, Tuple

def load_dataset(cfg: DictConfig) -> ds.Dataset:
    partitioned_dir = cfg.datasets.partitioned
    return ds.dataset(source=partitioned_dir, format="ipc", partitioning="hive")

def fold_loader(cfg: DictConfig) -> Iterator[Tuple[pa.Table, pa.Table]]:
    START_YEAR, END_YEAR = itemgetter('start', 'end')(cfg.years)
    dataset = load_dataset(cfg)
    for start_year, end_year in product([START_YEAR], range(START_YEAR+1, END_YEAR)):
        yield dataset.to_table(filter=((ds.field("year") >= ds.scalar(start_year)) & (ds.field("year") < ds.scalar(end_year)))), \
                dataset.to_table(filter=(ds.field("year") == ds.scalar(end_year)))

def load_everything(cfg: DictConfig) -> pa.Table:
    START_YEAR = itemgetter('start')(cfg.years)
    return load_dataset(cfg).to_table()

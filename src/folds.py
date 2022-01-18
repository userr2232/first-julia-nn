from omegaconf import DictConfig
from operator import itemgetter
import pyarrow as pa
import pyarrow.dataset as ds
from itertools import product
from typing import Iterator, Tuple


def fold_loader(cfg: DictConfig) -> Iterator[Tuple[pa.Table, pa.Table]]:
    START_YEAR, END_YEAR = itemgetter('start', 'end')(cfg.years)
    partitioned_dir = cfg.datasets.partitioned
    dataset = ds.dataset(source=partitioned_dir, format="ipc", partitioning="hive")
    for start_year, end_year in product([START_YEAR], range(START_YEAR+1, END_YEAR)):
        yield dataset.to_table(filter=((ds.field("year") >= ds.scalar(start_year)) & (ds.field("year") < ds.scalar(end_year)))), \
                dataset.to_table(filter=(ds.field("year") == ds.scalar(end_year)))
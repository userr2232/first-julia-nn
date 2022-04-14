from omegaconf import DictConfig
from operator import itemgetter
import pyarrow as pa
import pyarrow.dataset as ds
from itertools import product
from typing import Iterator, Tuple


def load_dataset(cfg: DictConfig) -> ds.Dataset:
    partitioned_dir = cfg.datasets.partitioned
    return ds.dataset(source=partitioned_dir, format="ipc", partitioning="hive")


def year_loader(cfg: DictConfig) -> Iterator[Tuple[Tuple, Tuple]]:
    START_YEAR, END_YEAR = itemgetter('start', 'end')(cfg.years)
    mode, training_window_length, validation_window_length = itemgetter('mode', 'training_window_length', 'validation_window_length')(cfg.cross_validation)
    
    training_start_year = START_YEAR
    if mode == "sliding_window":
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_start_year += 1
    elif mode == "expanding_window":
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_window_length += 1


def fold_loader(cfg: DictConfig) -> Iterator[Tuple[pa.Table, pa.Table]]:
    dataset = load_dataset(cfg)
    for (training_start_year, training_end_year), (validation_start_year, validation_end_year) in year_loader(cfg):
        yield dataset.to_table(filter=((ds.field("year") >= ds.scalar(training_start_year)) & (ds.field("year") < ds.scalar(training_end_year)))), \
                dataset.to_table(filter=((ds.field("year") >= ds.scalar(validation_start_year)) & (ds.field("year") < ds.scalar(validation_end_year))))


def load_everything(cfg: DictConfig) -> pa.Table:
    START_YEAR = itemgetter('start')(cfg.years)
    return load_dataset(cfg).to_table()

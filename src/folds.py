from omegaconf import DictConfig
from operator import itemgetter
import pyarrow as pa
import pyarrow.dataset as ds
from itertools import product
from typing import Iterator, Tuple


"""
    This function is used to load the dataset. The dataset must be partitioned.
    cfg: configuration file
    return: the dataset
"""
def load_dataset(cfg: DictConfig) -> ds.Dataset:
    partitioned_dir = cfg.datasets.partitioned
    return ds.dataset(source=partitioned_dir, format="ipc", partitioning="hive")

"""
    This function is used to load the folds for the cross validation.
    It uses two euristic methods to load the folds: sliding window and expanding window.
    For sliding window, the training window is moved one year at a time.
    For expanding window, the training window is increased one year at a time.
    cfg: configuration file
    return: a tuple with the training and validation datasets
"""
def year_loader(cfg: DictConfig) -> Iterator[Tuple[Tuple, Tuple]]:
    START_YEAR, END_YEAR = itemgetter('start', 'end')(cfg.years)
    # callable to get the mode, training window length and validation window length
    getter = itemgetter('mode', 'training_window_length', 'validation_window_length')
    mode, training_window_length, validation_window_length = getter(cfg.cross_validation)
    # equivalent to: 
    # mode = cfg.cross_validation.mode
    # training_window_length = cfg.cross_validation.training_window_length
    # validation_window_length = cfg.cross_validation.validation_window_length
    
    
    training_start_year = START_YEAR
    if mode == "sliding_window": # also known as rolling window
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            # TODO: check that there's no overlap between the training and validation windows
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_start_year += 1
    elif mode == "expanding_window":
        while training_start_year + training_window_length + validation_window_length <= END_YEAR:
            training_end_year = training_start_year + training_window_length
            yield (training_start_year, training_end_year), (training_end_year, training_end_year + validation_window_length)
            training_window_length += 1

"""
    This function is used to load the folds for the cross validation.
    It uses two euristic methods to load the folds: sliding window and expanding window.
    For sliding window, the training window is moved one year at a time.
    For expanding window, the training window is increased one year at a time.
    cfg: configuration file
    return: a tuple with the training and validation datasets
"""
def fold_loader(cfg: DictConfig) -> Iterator[Tuple[pa.Table, pa.Table]]:
    # Loads the dataset
    dataset = load_dataset(cfg)
    # Iterates through the splits. Each split is a tuple with the training and validation start and end years.
    for (training_start_year, training_end_year), (validation_start_year, validation_end_year) in year_loader(cfg):
        # Selects the data for the training and validation sets given the start and end years
        yield dataset.to_table(filter=((ds.field("year") >= ds.scalar(training_start_year)) & (ds.field("year") < ds.scalar(training_end_year)))), \
                dataset.to_table(filter=((ds.field("year") >= ds.scalar(validation_start_year)) & (ds.field("year") < ds.scalar(validation_end_year))))


"""
    Loads the entire dataset and returns it as a pyarrow table.
    cfg: configuration file
    return: the dataset as a pyarrow table
"""
def load_everything(cfg: DictConfig) -> pa.Table:
    return load_dataset(cfg).to_table()

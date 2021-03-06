from pathlib import Path
from typing import Union, Optional, Tuple
import pandas as pd
from omegaconf import DictConfig
from src.geomagindices import get_indices
import datetime
import torch
import logging
import joblib
import numpy as np
from operator import itemgetter
import time


INFERENCE_LOGGER_NAME = "INFERENCE"

def process_geoparam(cfg: DictConfig, now: datetime.datetime, hF: pd.DataFrame, ap: pd.DataFrame, f10_7: pd.DataFrame, DOY: int) -> Optional[torch.Tensor]:
    logger = logging.getLogger(INFERENCE_LOGGER_NAME)
    warning = lambda : logger.warning(f"No geoparameters available for this date range.")
    hF.dropna(inplace=True); hF.reset_index(drop=True, inplace=True)
    ap.dropna(inplace=True); ap.reset_index(drop=True, inplace=True)
    f10_7.dropna(inplace=True); f10_7.reset_index(drop=True, inplace=True)
    if len(hF) == 0 or len(ap) == 0 or len(f10_7) == 0:
        warning()
        return None

    _doy = 2 * np.pi * DOY / 365
    DNS, DNC = np.sin(_doy), np.cos(_doy)
    
    current_f10_7 = f10_7.f10_7.iloc[-1]
    if (now - f10_7.date.iloc[-1]).components.minutes > 30:
        warning()
        return None
    f10_7_90d = f10_7.f10_7.mean()

    current_ap = ap.ap.iloc[-1]
    ap_24h = ap.ap.mean()

    current_hF = hF.hF.iloc[-1]    
    prev_hF = hF.hF.iloc[0]

    scaler_path = Path(cfg.model.path) / cfg.model.scaler_checkpoint
    scaler = joblib.load(scaler_path)

    row = pd.DataFrame({'V_hF': current_hF, 'V_hF_prev': prev_hF, 'F10.7': current_f10_7, 'F10.7 (90d)': f10_7_90d,
                        'AP': current_ap, 'AP (24h)': ap_24h}, index=[0])

    row = scaler.transform(row)
    arr = np.append(row, [[DNS, DNC]], axis=-1)
    
    t = torch.from_numpy(arr).to(torch.float32).flatten()
    t.requires_grad = True

    return t


def predict(cfg: DictConfig, inputs: torch.Tensor) -> torch.Tensor:
    jit_module_path = Path(cfg.model.path) / cfg.model.nn_checkpoint
    model = torch.jit.load(jit_module_path)
    return model(inputs)


def save_prediction(cfg: DictConfig, nn_inputs: torch.Tensor, occurrence: bool, now: datetime.date) -> None:
    predictions_path, nn_inputs_path = itemgetter('predictions', 'nn_inputs')(cfg.datasets)
    df = None
    day_idx = (now - datetime.date(year=1970, month=1, day=1)).days
    row = []
    try:
        df = pd.read_csv(predictions_path)
    except FileNotFoundError:
        schema = [('day_idx', 'int'), ('prediction', 'int')]
        df = pd.DataFrame(np.empty(0, dtype=schema))
    l0 = len(df)
    row.append([day_idx, occurrence])
    df = pd.concat([df, pd.DataFrame(row, columns=['day_idx', 'prediction'])], ignore_index=True)
    df = df.sort_values('day_idx').drop_duplicates('day_idx', keep='last')
    if len(df) == l0 + 1:
        row = []
        inputs_df = pd.read_csv(nn_inputs_path)
        row.append([day_idx] + nn_inputs.tolist())
        inputs_df = pd.concat([inputs_df, 
                                pd.DataFrame(row, columns=['day_idx', 'V_hF', 'V_hF_prev', 'F10.7', 'F10.7 (90d)', 'AP', 'AP (24h)'])],
                                ignore_index=True)
        inputs_df = inputs_df.drop_duplicates('day_idx', keep='last')
        inputs_df.to_csv(nn_inputs_path, index=False)
    df.to_csv(predictions_path, index=False)


def daily_prediction(cfg: DictConfig, day: Optional[datetime.date] = None) -> None:
    tz = datetime.timezone(datetime.timedelta(hours=cfg.geomagneticindices.UTC_offset))
    if day is None:
        day = datetime.date.today()
    today = datetime.datetime.combine(day, datetime.time(), tzinfo=tz) + datetime.timedelta(hours=19, minutes=30)
    yesterday = today - datetime.timedelta(days=1)
    today_UTC = today.astimezone(datetime.timezone(datetime.timedelta(0))).replace(tzinfo=None)
    yesterday_UTC = today_UTC - datetime.timedelta(days=1)
    hF, ap, f10_7 = get_indices(cfg, pd.date_range(yesterday_UTC, today_UTC, freq="1D"))
    nn_inputs = process_geoparam(cfg, now=today.replace(tzinfo=None), hF=hF, ap=ap, f10_7=f10_7, DOY=today.timetuple().tm_yday)
    if nn_inputs is None: return
    nn_output = predict(cfg, inputs=nn_inputs)
    occurrence = torch.sigmoid(nn_output) >= 0.5
    print("occurrence", occurrence)
    save_prediction(cfg, nn_inputs=nn_inputs.detach().numpy()[:-2], occurrence=occurrence.item(), now=today.date())


def range_prediction(cfg: DictConfig) -> None:
    start, end = map(pd.Timestamp, itemgetter('start', 'end')(cfg.inference))
    date_range = pd.date_range(start, end, freq='1D')
    for date in date_range:
        daily_prediction(cfg, day=date)
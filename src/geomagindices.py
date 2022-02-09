import pandas as pd
from omegaconf import DictConfig
from operator import itemgetter
from ftplib import FTP
from pathlib import Path
import numpy as np
import io
from typing import Tuple


def download_hF(cfg: DictConfig, datetimes: pd.DatetimeIndex) -> pd.DataFrame:
    """Download and return hF for given dates and specified observation times
    Keyword arguments:
    cfg -- extra parameters specified in some .yaml file
    datetimes -- date range (freq=1D). It should not have obs times since these come from the cfg arg.
    """
    _datetimes = pd.date_range(datetimes[0].date(), datetimes[-1].date(), freq='1D')
    if _datetimes[0].year < 2020:
        raise NotImplementedError("This program supports fetching h'F from 2020 onward.")
    ftp_host, ftp_path = itemgetter('host', 'path')(cfg.geomagneticindices.hF)
    ftp_path = Path(ftp_path)
    ftp = FTP(ftp_host)
    ftp.login()
    station, obs_times = itemgetter('station', 'obs_times')(cfg.geomagneticindices.hF)
    schema = [('date', 'datetime64[ns]'), ('hF', 'float')]
    df = pd.DataFrame(np.empty(0, dtype=schema))
    rows = []
    for date in _datetimes:
        year, dayofyear = date.year, date.dayofyear
        dayofyear = str(dayofyear).zfill(3)
        extra_path = Path(itemgetter(year)(cfg.geomagneticindices.hF.year_mapping)) / dayofyear / "scaled"
        print("PATH:", str(ftp_path / extra_path))
        ftp.cwd(str(ftp_path / extra_path))
        for obs_time in obs_times:
            _date = date + pd.Timedelta(obs_time[:2]+'hours') + pd.Timedelta(obs_time[-2:]+'min')
            download_file = io.BytesIO()
            filename = f"{station}_{year}{dayofyear}{obs_time}00.SAO"
            ftp.retrbinary(f"RETR {filename}", download_file.write)
            download_file.seek(0)
            next = False
            hF = None
            for line in download_file.readlines():
                line = line.decode("utf-8")
                if next:
                    tmp = line[8 * 10: 8 * 11]
                    if tmp != "9999.000":
                        hF = float(tmp)
                    print("hF", hF)
                    break
                if line[:2] == "FF":
                    next = True
                    continue
            rows.append([_date, hF])
    df = pd.concat([df, pd.DataFrame(rows, columns=['date', 'hF'])], ignore_index=True)
    return df


def download_ap_f10_7(cfg: DictConfig, datetimes: pd.DatetimeIndex) -> pd.DataFrame:
    ftp_host, ftp_path = itemgetter('host', 'path')(cfg.geomagneticindices.Kp_ap_Ap_SN_F107)
    ftp = FTP(ftp_host)
    ftp.login()
    ftp.cwd(ftp_path)
    start_year, end_year = datetimes[0].year, datetimes[-1].year
    df = pd.DataFrame()
    for year in range(start_year, end_year+1):
        filename = f'Kp_ap_Ap_SN_F107_{year}.txt'
        download_file = io.BytesIO()
        ftp.retrbinary("RETR {}".format(filename), download_file.write)
        download_file.seek(0)
        colnames = ['YYYY', 'MM', 'DD', 'days', 'days_m', 'Bsr', 'dB', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'Ap', 'SN', 'F10.7obs', 'F10.7adj', 'D']
        df = pd.concat([df, pd.read_table(download_file, comment='#', delim_whitespace=True, header=None, names=colnames)], ignore_index=True)
    df.rename(columns={'YYYY': 'YEAR', 'MM': 'MONTH', 'DD': 'DAY', 'F10.7obs': 'f10_7'}, inplace=True)
    df['date'] = pd.to_datetime(df.loc[:, ('YEAR', 'MONTH', 'DAY')])
    df.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)
    aps = [f'ap{i}' for i in range(1, 9)]
    interm_df = df.drop(df.columns.difference(aps + ['date', 'f10_7']), axis=1).copy()
    schema = [('date', 'datetime64[ns]'), ('ap', 'int')]
    ap_df = pd.DataFrame(np.empty(0, dtype=schema))
    f10_7_df = pd.DataFrame(interm_df.loc[:, ['date', 'f10_7']]).copy()
    for _, row in interm_df.iterrows():
        date = row.date
        rows = []
        for ap in aps:
            rows.append([date, row[ap]])
            date = date + pd.Timedelta('3H')
        ap_df = pd.concat([ap_df, pd.DataFrame(rows, columns=['date', 'ap'])], ignore_index=True)
    return ap_df, f10_7_df


def get_indices(cfg: DictConfig, datetimes: pd.DatetimeIndex) -> Tuple[pd.DataFrame]:
    return download_hF(cfg, datetimes), download_ap_f10_7(cfg, datetimes)


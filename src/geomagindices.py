import pandas as pd
from omegaconf import DictConfig
from operator import itemgetter
from ftplib import FTP
from pathlib import Path
import numpy as np
import io
from typing import Tuple
import datetime


def download_hF(cfg: DictConfig, datetimes: pd.DatetimeIndex) -> pd.DataFrame:
    """Download and return hF for given dates and specified observation times
    Keyword arguments:
    cfg -- extra parameters specified in some .yaml file
    datetimes -- date range (freq=1D). It should not have obs times since these come from the cfg arg.
    """
    # gets all dates in the date range
    _datetimes = pd.date_range(datetimes[0].date(), datetimes[-1].date(), freq='1D')
    if _datetimes[0].year < 2020:
        raise NotImplementedError("This program supports fetching h'F from 2020 onward.")
    
    ftp_host, ftp_path = itemgetter('host', 'path')(cfg.geomagneticindices.hF)
    ftp_path = Path(ftp_path) 
    ftp = FTP(ftp_host)
    ftp.login() # anonymous login
    station, obs_times = itemgetter('station', 'obs_times')(cfg.geomagneticindices.hF)
    # Creates empty dataframe with the specified schema (date, hF)
    schema = [('date', 'datetime64[ns]'), ('hF', 'float')]
    df = pd.DataFrame(np.empty(0, dtype=schema))
    rows = []
    # iterates every day in the date range
    for date in _datetimes:
        year, dayofyear = date.year, date.dayofyear
        dayofyear = str(dayofyear).zfill(3) # pad with zeros
        extra_path = Path(itemgetter(year)(cfg.geomagneticindices.hF.year_mapping)) / dayofyear / "scaled"
        try:
            ftp.cwd(str(ftp_path / extra_path))
        except:
            continue
        for obs_time in obs_times:
            _date = date + pd.Timedelta(obs_time[:2]+'hours') + pd.Timedelta(obs_time[-2:]+'min')
            download_file = io.BytesIO()
            filename = f"{station}_{year}{dayofyear}{obs_time}00.SAO"
            # TODO: check what SAO version is in the server
            # TODO: document the following code
            try:
                ftp.retrbinary(f"RETR {filename}", download_file.write)
            except:
                continue
            download_file.seek(0)
            next = False
            hF = None
            for line in download_file.readlines():
                line = line.decode("utf-8")
                if next:
                    tmp = line[8 * 10: 8 * 11]
                    if tmp != "9999.000":
                        hF = float(tmp)
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
    df = pd.DataFrame()
    colnames = ['YYYY', 'MM', 'DD', 'days', 'days_m', 'Bsr', 'dB', 'Kp1', 'Kp2', 'Kp3', 'Kp4', 'Kp5', 'Kp6', 'Kp7', 'Kp8', 'ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'ap6', 'ap7', 'ap8', 'Ap', 'SN', 'F10.7obs', 'F10.7adj', 'D']
    real_time = (datetime.datetime.now() - datetimes[0]).days <= 2
    start_year, end_year = datetimes[0].year, datetimes[-1].year
    for year in range(start_year-1, end_year+1):
        filename = f'Kp_ap_Ap_SN_F107_{year}.txt'
        download_file = io.BytesIO()
        ftp.retrbinary("RETR {}".format(filename), download_file.write)
        download_file.seek(0)
        df = pd.concat([df, pd.read_table(download_file, comment='#', delim_whitespace=True, header=None, names=colnames)], ignore_index=True)
    
    filename = 'Kp_ap_Ap_SN_F107_nowcast.txt'
    download_file = io.BytesIO()
    ftp.retrbinary("RETR {}".format(filename), download_file.write)
    download_file.seek(0)
    # combine yearly data with more recent data
    df = pd.concat([df, pd.read_table(download_file, comment='#', delim_whitespace=True, header=None, names=colnames)], ignore_index=True)

    df.rename(columns={'YYYY': 'YEAR', 'MM': 'MONTH', 'DD': 'DAY', 'F10.7obs': 'f10_7'}, inplace=True)
    df['date'] = pd.to_datetime(df.loc[:, ('YEAR', 'MONTH', 'DAY')])
    # sort by date and drop repeated rows
    df = df.sort_values('date').drop_duplicates('date',keep='last')

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


"""
    Gets the data from the two FTP servers and returns it as dataframes.
    The geomagnetic indices are returned in the following order: hF, ap, f10_7. These are h'F, ap and F10.7 respectively.
"""
def get_indices(cfg: DictConfig, date_range: pd.DatetimeIndex) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ap, f10_7 = download_ap_f10_7(cfg, date_range)
    ap.replace(-1, np.nan, inplace=True)
    f10_7.replace(-1, np.nan, inplace=True)
    hF = download_hF(cfg, date_range)
    ap = ap.loc[((ap.date >= date_range[0]) & (ap.date <= date_range[-1]))].copy().reset_index(drop=True)
    f10_7 = f10_7.loc[((f10_7.date >= (date_range[0] - pd.Timedelta("90D"))) & (f10_7.date <= date_range[-1]))].copy().reset_index(drop=True)
    hF = hF.loc[((hF.date >= date_range[1] - datetime.timedelta(minutes=30)) & (hF.date <= date_range[-1]))].copy().reset_index(drop=True)

    delta = pd.Timedelta(f"{cfg.geomagneticindices.UTC_offset}h")
    ap.date = ap.date + delta
    f10_7.date = f10_7.date + delta
    hF.date = hF.date + delta
    return hF, ap, f10_7

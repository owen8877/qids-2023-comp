from unittest import TestCase

import pandas as pd
from pandas import Series, DataFrame
from xarray import Dataset

import pipeline
from util import ensure_dir


def parse_date_time_column(s: Series):
    """
    Parse the date_time field and scatter into asset/day/timeslot.

    :param s: a `Series` object where the entries follow `s[asset]d[day]p[timeslot]`
    :return: the parsed dataframe
    """
    return s.str.split(pat='s|d|p', expand=True).iloc[:, 1:].astype(int).rename(
        columns={1: 'asset', 2: 'day', 3: 'timeslot'})


def parse_date_column(s: Series):
    """
    Parse the date_time field and scatter into asset/day.

    :param s: a `Series` object where the entries follow `s[asset]d[day]`
    :return: the parsed dataframe
    """
    return s.str.split(pat='s|d', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day'})


def pre_process_df_with_date_time_legacy(df: DataFrame):
    date_time_series = df['date_time']
    p_df = pd.concat((parse_date_time_column(date_time_series), df.drop(columns='date_time')), axis=1)
    return p_df.set_index(['day', 'asset', 'timeslot']).sort_index()


def pre_process_df_with_date_legacy(df: DataFrame):
    date_time_series = df['date_time']
    p_df = pd.concat((parse_date_column(date_time_series), df.drop(columns='date_time')), axis=1)
    return p_df.set_index(['day', 'asset']).sort_index()


def pre_process_df(f_df: DataFrame, m_df: DataFrame):
    f_ds = Dataset.from_dataframe(pre_process_df_with_date_legacy(f_df))
    m_ds = Dataset.from_dataframe(pre_process_df_with_date_time_legacy(m_df))
    return f_ds.merge(m_ds)


def _dump(is_mini: bool = False, n_days: int = 10, path_prefix='..', nc_destination: str = 'nc',
          raw_source: str = 'raw', is_first_round: bool = True):
    """
    The actual working file that dumps the dataset file.

    :param is_mini:
    :param n_days:
    :return:
    """
    NC_PATH = f'data/{nc_destination}'
    RAW_PATH = f'data/{raw_source}'

    nc_path = f'{path_prefix}/{NC_PATH}' + (f'_mini/{n_days}' if is_mini else '')
    raw_path = f'{path_prefix}/{RAW_PATH}' + (f'_mini/{n_days}' if is_mini else '')
    template = '{}/' + ('first' if is_first_round else 'second') + '_round_train_{}_data.csv'
    ensure_dir(nc_path)

    try:
        f_df = pre_process_df_with_date_legacy(pd.read_csv(template.format(raw_path, 'fundamental')))
        m_df = pre_process_df_with_date_time_legacy(pd.read_csv(template.format(raw_path, 'market')))
        r_df = pre_process_df_with_date_legacy(pd.read_csv(template.format(raw_path, 'return')))
        ds = Dataset.from_dataframe(f_df)
        ds.update(Dataset.from_dataframe(m_df))
        ds.update(Dataset.from_dataframe(r_df))
        ds.to_netcdf(f'{nc_path}/base.nc')
    except FileNotFoundError as e:
        print('csv raw data not found! make sure that the training fundamental/market/return data '
              f'is under the @{raw_path} folder')
        raise e


def _dump_legacy(is_mini: bool = False, n_days: int = 10, path_prefix='..'):
    """
    The actual working file that dumps the dataset file.

    :param is_mini:
    :param n_days:
    :return:
    """
    PARSED_PATH = 'data/parsed'
    RAW_PATH = 'data/raw'

    parsed_path = f'{path_prefix}/{PARSED_PATH}' + (f'_mini/{n_days}' if is_mini else '')
    raw_path = f'{path_prefix}/{RAW_PATH}' + (f'_mini/{n_days}' if is_mini else '')
    template = '{}/first_round_train_{}_data.csv'
    try:
        f_df = pre_process_df_with_date_legacy(pd.read_csv(template.format(raw_path, 'fundamental')))
        m_df = pre_process_df_with_date_time_legacy(pd.read_csv(template.format(raw_path, 'market')))
        r_df = pre_process_df_with_date_legacy(pd.read_csv(template.format(raw_path, 'return')))
        pipeline.Dataset(m_df, f_df, r_df).dump(parsed_path)
    except FileNotFoundError as e:
        print('csv raw data not found! make sure that the training fundamental/market/return data '
              f'is under the @{raw_path} folder')
        raise e


class Test(TestCase):
    def test_parse_raw_df_and_dump_full(self):
        _dump(is_mini=False, n_days=-1)

    def test_parse_raw_df_and_dump_second_round(self):
        _dump(is_mini=False, n_days=-1, nc_destination='nc_2round', raw_source='second_round_datasets',
              is_first_round=False)

    def test_parse_raw_df_and_dump_mini(self, n_days: int = 10):
        _dump(is_mini=True, n_days=n_days)

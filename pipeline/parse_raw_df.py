from unittest import TestCase

import pandas as pd
from pandas import Series, DataFrame

import pipeline


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


def pre_process_df_with_date_time(df: DataFrame):
    date_time_series = df['date_time']
    p_df = pd.concat((parse_date_time_column(date_time_series), df.drop(columns='date_time')), axis=1)
    return p_df.set_index(['day', 'asset', 'timeslot']).sort_index()


def pre_process_df_with_date(df: DataFrame):
    date_time_series = df['date_time']
    p_df = pd.concat((parse_date_column(date_time_series), df.drop(columns='date_time')), axis=1)
    return p_df.set_index(['day', 'asset']).sort_index()


def _dump(is_mini: bool = False, n_days: int = 10):
    """
    The actual working file that dumps the dataset file.

    :param is_mini:
    :param n_days:
    :return:
    """
    parsed_path = f'../data/parsed_mini/{n_days}' if is_mini else '../data/parsed'
    raw_path = f'../data/raw_mini/{n_days}' if is_mini else '../data/raw'
    template = '{}/first_round_train_{}_data.csv'
    try:
        f_df = pre_process_df_with_date(pd.read_csv(template.format(raw_path, 'fundamental')))
        m_df = pre_process_df_with_date_time(pd.read_csv(template.format(raw_path, 'market')))
        r_df = pre_process_df_with_date(pd.read_csv(template.format(raw_path, 'return')))
        pipeline.Dataset(m_df, f_df, r_df).dump(parsed_path)
    except FileNotFoundError as e:
        print('csv raw data not found! make sure that the training fundamental/market/return data '
              f'is under the @{raw_path} folder')
        raise e


class Test(TestCase):
    def test_parse_raw_df_and_dump_full(self):
        _dump(is_mini=False, n_days=-1)

    def test_parse_raw_df_and_dump_mini(self, n_days: int = 10):
        _dump(is_mini=True, n_days=n_days)

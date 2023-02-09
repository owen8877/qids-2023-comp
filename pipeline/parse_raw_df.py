from unittest import TestCase

import pandas as pd
from pandas import Series, DataFrame

from pipeline import Dataset


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
    return pd.concat((parse_date_time_column(date_time_series), df.drop(columns='date_time')), axis=1)


def pre_process_df_with_date(df: DataFrame):
    date_time_series = df['date_time']
    return pd.concat((parse_date_column(date_time_series), df.drop(columns='date_time')), axis=1)


class Test(TestCase):
    def test_parse_raw_df_and_dump(self):
        f_df = pre_process_df_with_date(pd.read_csv('../data/raw/first_round_train_fundamental_data.csv'))
        m_df = pre_process_df_with_date_time(pd.read_csv('../data/raw/first_round_train_market_data.csv'))
        r_df = pre_process_df_with_date(pd.read_csv('../data/raw/first_round_train_return_data.csv'))

        Dataset(m_df, f_df, r_df).dump('../data/parsed')

# Basic packages to import (NO NEED TO ADJUST IT)
import numpy as np
import pandas as pd
import xarray as xr
from pandas import DataFrame, Series
from xarray import Dataset

# The addresses for the train data inputs (NO NEED TO ADJUST IT)
from config import Path

historical_data_path = Path.historical_data_path
MARKET_DATA_PATH = f'{historical_data_path}/second_round_train_market_data.csv'
FUNDAMENTAL_DATA_PATH = f'{historical_data_path}/second_round_train_fundamental_data.csv'
RETURN_DATA_PATH = f'{historical_data_path}/second_round_train_return_data.csv'


class Portfolio:
    def __init__(self):
        self.lookback_window = 1

    def initialize(self, X, y):
        pass

    def train(self, X, y):
        pass

    def construct(self, X):
        current_day = X.day.max().item()
        # X_ = X.sel(day=slice(current_day-self.lookback_window, current_day))
        X_ = X.sel(day=[current_day])
        pe = X_.pe
        good_company = (pe > 0) & (pe < 15)  # find companies with good pe
        market_pe = pe.median().item()  # find market median pe to see if market is hyped
        investment = np.where(good_company, 1 / pe ** 5, 0)  # invest only the good company
        s = max(1e-6, investment.sum())
        y = investment / s
        return y * min(50 / market_pe, 1)  # scaled by market hypeness


INITIALIZED: bool = False
DS: Dataset = None
PORTFOLIO: Portfolio = None


def parse_date_column(s: Series):
    """
    Parse the date_time field and scatter into asset/day.

    :param s: a `Series` object where the entries follow `s[asset]d[day]`
    :return: the parsed dataframe
    """
    return s.str.split(pat='s|d', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day'})


def parse_date_time_column(s: Series):
    """
    Parse the date_time field and scatter into asset/day/timeslot.

    :param s: a `Series` object where the entries follow `s[asset]d[day]p[timeslot]`
    :return: the parsed dataframe
    """
    return s.str.split(pat='s|d|p', expand=True).iloc[:, 1:].astype(int).rename(
        columns={1: 'asset', 2: 'day', 3: 'timeslot'})


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



def load_historical_data():
    """
    loadign the data file
    :return:
    """
    global DS

    f_df = pre_process_df_with_date_legacy(pd.read_csv(FUNDAMENTAL_DATA_PATH))
    m_df = pre_process_df_with_date_time_legacy(pd.read_csv(MARKET_DATA_PATH))
    r_df = pre_process_df_with_date_legacy(pd.read_csv(RETURN_DATA_PATH))

    DS = Dataset.from_dataframe(f_df)
    DS.update(Dataset.from_dataframe(m_df))
    DS.update(Dataset.from_dataframe(r_df))


# Important: Get your decision output
def get_decisions(market_df: DataFrame, fundamental_df: DataFrame):
    global DS, INITIALIZED, PORTFOLIO
    feature = ['pe']
    if not INITIALIZED:
        load_historical_data()
        PORTFOLIO = Portfolio()
        PORTFOLIO.initialize(DS[feature], None)
        INITIALIZED = True

    # Step 1: parse and process dataframe
    new_ds = pre_process_df(fundamental_df, market_df)
    new_ds['return'] = new_ds['close']
    DS = xr.concat((DS, new_ds), dim='day')

    # Step 2: get desired data
    DS_current = DS[feature].isel(day=range(-PORTFOLIO.lookback_window, 0))
    return_current = DS['return'].isel(day=range(-PORTFOLIO.lookback_window, 0))

    # Step 3: run Portfolio and get decision
    # portfolio = PEPortfolio()

    # Store the decisions here
    decision_list = PORTFOLIO.construct(DS_current).squeeze()

    # ds[features].sel(day=slice(chunk_start_day, current_day))
    # tr = portfolio.construct(sub_ds)
    ###################################################################################################################
    # TODO: Write your code here

    ###################################################################################################################

    # Output the decision at this moment
    return decision_list

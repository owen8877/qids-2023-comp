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


def extract_market_data(ds: Dataset):
    """
    Input the market data and extract mean price using close
    prices, volatility, daily return and mean volume
    :param ds: [xr.Dataset] dataset that contains market data
    :return: [xr.Dataset] extracted features from market data
    """
    assert not ds.isnull().any().to_array().any().item()

    day_money_sum = ds['money'].sum('timeslot')
    day_volume_sum = ds['volume'].sum('timeslot')
    avg_price = day_money_sum / day_volume_sum
    day_volume_s = day_volume_sum.to_series()
    indx_day = day_volume_s.where(day_volume_s == 0).dropna()

    for day_idx, asset_idx in indx_day.index:
        sub_ds = ds.sel(day=day_idx, asset=asset_idx)
        replace_value = (sub_ds['high'].max() + sub_ds['low'].min()) / 2
        avg_price.loc[dict(day=day_idx, asset=asset_idx)] = replace_value

    assert not avg_price.isnull().any().item(), "Clean data still contains NaN"

    # Compute volatility
    T = len(ds.timeslot)  # number of time units
    # note numpy use 0 dof while pd use 1 dof
    volatility = ds['close'].std('timeslot', ddof=1) * np.sqrt(T)

    # Compute average volume:
    mean_volume = day_volume_sum / T

    # Daily return that compares the first open and the last close
    daily_close_price = ds['close'].sel(timeslot=T, drop=True)
    daily_open_price = ds['open'].sel(timeslot=1, drop=True)
    daily_high_price = ds['high'].max(dim='timeslot')
    daily_low_price = ds['low'].min(dim='timeslot')

    previous_close_price = daily_close_price.shift(day=1)
    previous_close_price[dict(day=0)] = daily_open_price.isel(day=0)
    daily_return = daily_close_price / previous_close_price - 1

    return Dataset(dict(
        avg_price=avg_price,
        volatility=volatility,
        mean_volume=mean_volume,
        close_0=daily_close_price,
        open_0=daily_open_price,
        high_0=daily_high_price,
        low_0=daily_low_price,
        return_0=daily_return,
    ))



class Portfolio:
    def __init__(self):
        self.lookback_window = 100
        pass

    def initialize(self, X, y):
        pass

    def train(self, X, y):
        pass


    def prepare_feature(self, ds):
        ds['return_0'] = ds['close_0'] / ds['close_0'].shift(day=1) - 1
        ds['close_moving_5'] = ds['close_0'].rolling(day=5).mean()
        ds['close_moving_20'] = ds['close_0'].rolling(day=20).mean()
        ds['close_moving_100'] = ds['close_0'].rolling(day=100).mean()
        ds['close_ma_diff_5'] = ds['close_0'] - ds['close_moving_5']
        ds['close_ma_diff_20'] = ds['close_0'] - ds['close_moving_20']
        ds['close_ma_diff_100'] = ds['close_0'] - ds['close_moving_100']
        ds['pe_moving_diff_5'] = ds['pe'].median(dim='asset') - ds['pe'].median(dim='asset').rolling(day=5).mean()
        ds['pe_moving_diff_10'] = ds['pe'].median(dim='asset') - ds['pe'].median(dim='asset').rolling(day=10).mean()
        return ds

    def construct(self, X):
        X = self.prepare_feature(X.copy())

        current_day = X.day.max().item()
        X_ = X.sel(day=[current_day])
        pe = X_.pe
        bear_thr = -0.05
        bull_thr = 0.05
        market_pe = pe.median().item()
        is_bear = X_['pe_moving_diff_5'] < X_['pe_moving_diff_10']

        dont_buy = (X_.return_0 < bear_thr) & (X_.close_ma_diff_20 < 0) & (X_.close_ma_diff_100 < 0)

        good_company = (pe > 0) & (pe < 15) & (1 - dont_buy)
        investment = np.where(good_company, pe ** (-5), 0)
        # print(investment.max())
        # add stocks with bull performance when market is good (strategy one)
        # version_1
        # investment += (market_pe<40)*(X_.close_ma_diff_5>0)*(X_.close_ma_diff_20>0)*(X_.close_ma_diff_100>0)\
        #               *(X_.return_0>bull_thr)*(X_.return_0-bull_thr)/(X_.return_0-bull_thr).max()*investment.max()
        # version_2:
        # investment += (market_pe<40)*(1-is_bear)*(X_.close_ma_diff_5>0)*(X_.close_ma_diff_20>0)*(X_.close_ma_diff_100<0)\
        #               *(X_.return_0>bull_thr)*(X_.return_0-bull_thr)/(X_.return_0-bull_thr).max()*investment.max()*0.5

        s = max(1e-6, investment.sum())
        y = investment / s

        # strategy 2
        propose = (y * (
                    min(50 / market_pe, 1) * (1 - is_bear).values + is_bear.values * min(market_pe / 25, 1))).squeeze()

        # propose = (y* (min(50 / market_pe, 1))).squeeze()

        propose[propose < 0.01] = 0
        # print(current_day, np.where(propose > 0), propose[np.where(propose > 0)])
        return propose

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
    summary_ds = extract_market_data(m_ds)
    return f_ds.merge(m_ds).merge(summary_ds)



def load_historical_data():
    """
    loadign the data file
    :return:
    """
    global DS

    f_df = pre_process_df_with_date_legacy(pd.read_csv(FUNDAMENTAL_DATA_PATH))
    m_df = pre_process_df_with_date_time_legacy(pd.read_csv(MARKET_DATA_PATH))
    summary_ds = extract_market_data(Dataset.from_dataframe(m_df))
    r_df = pre_process_df_with_date_legacy(pd.read_csv(RETURN_DATA_PATH))

    DS = Dataset.from_dataframe(f_df)
    DS.update(Dataset.from_dataframe(m_df))
    DS.update(summary_ds)
    DS.update(Dataset.from_dataframe(r_df))


# Important: Get your decision output
def get_decisions(market_df: DataFrame, fundamental_df: DataFrame):
    global DS, INITIALIZED, PORTFOLIO
    # feature = ['pe',  'close_ma_diff_5', 'pe_moving_diff_5', 'pe_moving_diff_10', 'close_ma_diff_20', 'close_ma_diff_100', 'return_0']
    feature = ['pe', 'close_0']
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
    # return_current = DS['return'].isel(day=range(-PORTFOLIO.lookback_window, 0))

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
    return list(decision_list)

import pandas as pd
from pandas import Series, MultiIndex
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from joblib import dump, load

from qids_lib import make_env

__num_of_stocks = 54
__point_per_day = 50

env = make_env()


def parse_date_time_column(s: Series):
    return s.str.split(pat='s|d|p', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day', 3: 'timeslot'})


def parse_date_column(s: Series):
    return s.str.split(pat='s|d', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day'})


def pre_process_df_with_date_time(df):
    date_time_series = df['date_time']
    df = df.drop(columns='date_time')
    df.index = MultiIndex.from_frame(parse_date_time_column(date_time_series))
    return df


def pre_process_df_with_date(df):
    date_series = df['date_time']
    df = df.drop(columns='date_time')
    df.index = MultiIndex.from_frame(parse_date_column(date_series))
    return df


model: LinearRegression = load('model/linear/param.joblib')

while not env.is_end():
    f_df, m_df = env.get_current_market()

    m_df = pre_process_df_with_date_time(m_df)
    f_df = pre_process_df_with_date(f_df)

    m_agg_df = m_df.groupby(level=[0, 1]).mean().sort_index()
    full_df = pd.concat([m_agg_df, f_df], axis=1).dropna()

    X = full_df[['close', 'volume', 'money', 'turnoverRatio', 'transactionAmount', 'pe_ttm', 'pcf']]
    X_last_day = X.iloc[X.index.get_level_values(1) == X.index.get_level_values(1).max()].sort_index()
    y = model.predict(X_last_day)

    env.input_prediction(Series(data=y, index=X.index))


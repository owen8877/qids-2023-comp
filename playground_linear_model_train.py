import pandas as pd
from pandas import Series, MultiIndex
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from joblib import dump

__num_of_stocks = 54
__point_per_day = 50

f_df = pd.read_csv('data/first_round_train_fundamental_data.csv')
m_df = pd.read_csv('data/first_round_train_market_data.csv')
r_df = pd.read_csv('data/first_round_train_return_data.csv')


def parse_date_time_column(s: Series):
    return s.str.split(pat='s|d|p', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day', 3: 'timeslot'})


def parse_date_column(s: Series):
    return s.str.split(pat='s|d', expand=True).iloc[:, 1:].astype(int).rename(columns={1: 'asset', 2: 'day'})


def pre_process_df_with_date_time(df):
    date_time_series = df['date_time']
    df.drop(columns='date_time', inplace=True)
    df.index = MultiIndex.from_frame(parse_date_time_column(date_time_series))


def pre_process_df_with_date(df):
    date_series = df['date_time']
    df.drop(columns='date_time', inplace=True)
    df.index = MultiIndex.from_frame(parse_date_column(date_series))


pre_process_df_with_date_time(m_df)
pre_process_df_with_date(r_df)
pre_process_df_with_date(f_df)

m_agg_df = m_df.groupby(level=[0, 1]).mean().sort_index()
full_df = pd.concat([m_agg_df, f_df, r_df], axis=1).dropna()

X = full_df[['close', 'volume', 'money', 'turnoverRatio', 'transactionAmount', 'pe_ttm', 'pcf']]
y = full_df['return']

kf = KFold(n_splits=5, shuffle=True, random_state=10)
for train, test in kf.split(full_df.index):
    reg = LinearRegression().fit(X.iloc[train], y.iloc[train])
    train_score = reg.score(X.iloc[train], y.iloc[train])
    test_score = reg.score(X.iloc[test], y.iloc[test])
    print(f'train score: {train_score:.4f}, test score: {test_score:.4f}')

final_model = LinearRegression().fit(X, y)
dump(final_model, 'model/linear/param.joblib')
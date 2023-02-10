import pandas as pd
from joblib import load
from pandas import Series
from sklearn.linear_model import LinearRegression

from pipeline.parse_raw_df import pre_process_df_with_date_time, pre_process_df_with_date
from qids_lib import make_env

__num_of_stocks = 54
__point_per_day = 50

env = make_env()

model: LinearRegression = load('../../model/linear/param.joblib')

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

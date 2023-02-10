import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from pipeline import Dataset

dataset = Dataset.load()
m_agg_df = pd.concat([
    dataset.market.set_index('timeslot').loc[1].set_index(['asset', 'day'])['open'],
    dataset.market.set_index('timeslot').loc[50].set_index(['asset', 'day'])['close'],
    dataset.market.groupby(['asset', 'day']).agg(high=('high', max), low=('low', min), var=('close', np.std),
                                                 avg=('close', np.mean), volume=('volume', np.sum),
                                                 money=('money', np.sum)),
], axis=1)
full_df = pd.concat(
    [m_agg_df, dataset.fundamental.set_index(['asset', 'day']), dataset.ref_return.set_index(['asset', 'day'])],
    axis=1).dropna()


# construct feature
def get_single_day_return(s):
    return (s.shift(-1) / s - 1).fillna(0)


wdf = dataset.market.groupby(by='asset').apply(
    lambda df: get_single_day_return(df.set_index('timeslot').loc[50].set_index('day')['close']))
hold_1_return = wdf.reset_index().melt(id_vars=['asset'], value_vars=wdf.columns,
                                       value_name='hold_1_return').sort_values(by=['asset', 'day']).reset_index(
    drop=True)
day_before_hold_1_return = hold_1_return.set_index(['asset', 'day']).groupby(level=0).shift(1, fill_value=0)
full_df['hold_1_return'] = hold_1_return.set_index(['asset', 'day'])
full_df['day_before_hold_1_return'] = day_before_hold_1_return

full_df_swap = full_df.swaplevel().sort_index()
X = full_df_swap[['close', 'volume', 'money', 'std', 'turnoverRatio', 'transactionAmount', 'pe_ttm', 'pcf']]
# X = full_df[['day_before_hold_1_return']]
y = full_df_swap['return']

tscv = TimeSeriesSplit(n_splits=5)
for train, test in tscv.split(full_df_swap):
    reg = LinearRegression().fit(X.iloc[train], y.iloc[train])
    train_score = reg.score(X.iloc[train], y.iloc[train])
    test_score = reg.score(X.iloc[test], y.iloc[test])
    print(f'train score: {train_score:.4f}, test score: {test_score:.4f}')

final_model = LinearRegression().fit(X, y)
# dump(final_model, 'model/linear/param.joblib')

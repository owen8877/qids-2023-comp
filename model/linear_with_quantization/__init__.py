import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression

from datatools import extract_market_data, data_quantization
from pipeline import Dataset
from pipeline.backtest import evaluation_for_submission, SupportsPredict
from visualization.metric import plot_performance

dataset = Dataset.load('../data/parsed')
df = pd.concat([dataset.fundamental, extract_market_data(dataset.market)], axis=1).dropna()
f_quantile_feature = ['turnoverRatio_QUANTILE', 'transactionAmount_QUANTILE', 'pb_QUANTILE', 'ps_QUANTILE',
                      'pe_ttm_QUANTILE', 'pe_QUANTILE', 'pcf_QUANTILE']
m_quantile_feature = ['avg_price_QUANTILE', 'volatility_QUANTILE', 'mean_volume_QUANTILE']
feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe_ttm', 'pe', 'pcf', 'avg_price', 'volatility',
           'mean_volume']


def linear_model(X: DataFrame, y: Series) -> SupportsPredict:
    reg = LinearRegression().fit(X, y)
    return reg


q_df, _ = data_quantization(dataset.fundamental)
full_df = pd.concat([q_df, dataset.ref_return], axis=1).dropna()
model = linear_model(full_df[f_quantile_feature], full_df['return'])

performance = evaluation_for_submission(model, f_quantile_feature, dataset=dataset, df=df, lookback_window=None)

plt.figure()
plot_performance(performance, metrics_selected=['train_r2', 'test_cum_r2', 'test_cum_pearson'])
plt.show()

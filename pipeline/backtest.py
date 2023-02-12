from typing import Union, Tuple, List, Callable, Any
from unittest import TestCase

import pandas as pd
from pandas import Series, DataFrame
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm

from pipeline import Dataset


def cross_validation(training: Callable[[DataFrame, Series], Any],
                     feature_columns: Union[List[str], Tuple[str]],
                     df: DataFrame = None, n_splits: int = 997, return_column: str = 'return',
                     lookback_window: int = None):
    """
    Perform cross validation backtest on the given set of features.

    :param training: a function that builds a model based on the input feature/target arguments.
            Signature: (DataFrame, Series) -> Model (that supports `.predict()` method)
    :param feature_columns:
    :param df: the full dataframe containing all necessary data
    :param n_splits: number of splits, 997 by default (since the return data is missing on day 999 and 1000, and we need
            the first day to be in the first training fold)
    :param return_column:
    :param lookback_window: max number of days of the training feature dataframe, `None` means no truncation.
    :return: performance: a dictionary containing the metric evaluation for each fold.
    """
    if df is None:
        dataset = Dataset.load(f'{__file__}/../data/parsed')
        df = pd.concat(
            [dataset.fundamental.set_index(['asset', 'day']), dataset.ref_return.set_index(['asset', 'day'])],
            axis=1).dropna()

    if df.index.names != ['day', 'asset']:
        df = df.swaplevel().sort_index()

    days = df.index.get_level_values('day').unique()
    if len(days) < n_splits:
        print('Warning: number of splits is larger than days available in the training set.')
    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(days)))
    iterator = tqdm(tscv.split(days), total=tscv.n_splits)
    cum_y_val_true = Series(dtype=float)
    cum_y_val_prediction = Series(dtype=float)
    performance = []

    for train, val in iterator:
        # X, _ = data_quantization(df[original_feature])

        days_train = days[train]
        if (lookback_window is not None) and (len(days_train) > lookback_window):
            days_train = days_train[-lookback_window:]
        X_train, y_train_true = df.loc[(days_train,), :][feature_columns], df.loc[(days_train,), :][return_column]
        model = training(X_train, y_train_true)
        y_train_prediction = Series(model.predict(X_train), index=y_train_true.index)

        days_val = days[val]
        X_val, y_val_true = df.loc[(days_val,), :][feature_columns], df.loc[(days_val,), :][return_column]
        y_val_prediction = Series(model.predict(X_val), index=y_val_true.index)

        cum_y_val_true = pd.concat([cum_y_val_true, y_val_true])
        cum_y_val_prediction = pd.concat([cum_y_val_prediction, y_val_prediction])

        performance.append({
            'train_r2': r2_score(y_train_true, y_train_prediction),
            'val_r2': r2_score(y_val_true, y_val_prediction),
            'val_pearson': y_val_true.corr(y_val_prediction),
            'val_cum_pearson': cum_y_val_true.corr(cum_y_val_prediction),
        })

    return performance


class Test(TestCase):
    def test_cross_validation(self):
        """
        A minimalist set-up to show how to use the backtest cross-validation suite.
        """
        from sklearn.linear_model import LinearRegression
        from visualize.metric import plot_performance
        from matplotlib import pyplot as plt
        from datatools import data_quantization

        dataset = Dataset.load('../data/parsed')
        quantized_fundamental, _ = data_quantization(dataset.fundamental.set_index(['asset', 'day']))
        df = pd.concat(
            [quantized_fundamental, dataset.fundamental.set_index(['asset', 'day']),
             dataset.ref_return.set_index(['asset', 'day'])], axis=1).dropna()
        quantile_feature = ['turnoverRatio_QUANTILE', 'transactionAmount_QUANTILE', 'pb_QUANTILE', 'ps_QUANTILE',
                            'pe_ttm_QUANTILE', 'pe_QUANTILE', 'pcf_QUANTILE']
        original_feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe_ttm', 'pe', 'pcf']

        def linear_model(X: DataFrame, y: Series):
            reg = LinearRegression().fit(X, y)
            return reg

        performance = cross_validation(linear_model, quantile_feature, df=df, n_splits=997, lookback_window=200)

        plt.figure()
        plot_performance(performance, metrics_selected=['train_r2', 'val_cum_pearson'])
        plt.show()

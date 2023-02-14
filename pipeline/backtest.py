from abc import abstractmethod
from typing import Union, Tuple, List, Callable, Iterable
from typing import runtime_checkable, Protocol  # ERASE_MAGIC
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm.auto import tqdm, trange

from datatools import extract_market_data, data_quantization
from pipeline import Dataset, N_train_days, N_asset, N_timeslot, N_test_days
from pipeline.parse_raw_df import pre_process_df_with_date_time, pre_process_df_with_date
from qids_lib import QIDS
from visualization.metric import Performance


@runtime_checkable  # ERASE_MAGIC
class SupportsPredict(Protocol):  # ERASE_MAGIC
    """An ABC with one abstract method `predict`. # ERASE_MAGIC"""
    __slots__ = ()  # ERASE_MAGIC

    @abstractmethod  # ERASE_MAGIC
    def predict(self, X: DataFrame) -> Series:  # ERASE_MAGIC
        pass  # ERASE_MAGIC


ModelLike = Union[Callable[[DataFrame, Series], SupportsPredict], SupportsPredict]  # ERASE_MAGIC


def cross_validation(training: ModelLike, feature_columns: Union[List[str], Tuple[str]], df: DataFrame = None,
                     n_splits: int = 997, return_column: str = 'return', lookback_window: int = None) -> Performance:
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
        df = pd.concat([dataset.fundamental, dataset.ref_return], axis=1).dropna()

    if df.index.names != ['day', 'asset']:  # will be replaced by a generic dataframe checker
        raise Exception(f'The index level of df should be `[day, asset]`, got {df.index.names} instead!')

    days = df.index.get_level_values('day').unique()
    if len(days) < n_splits:
        print('Warning: number of splits is larger than days available in the training set.')
    tscv = TimeSeriesSplit(n_splits=min(n_splits, len(days)))
    pbar = tqdm(tscv.split(days), total=tscv.n_splits)
    cum_y_val_true = Series(dtype=float)
    cum_y_val_prediction = Series(dtype=float)
    performance = Performance()

    for fold, (train, val) in enumerate(pbar):
        # X, _ = data_quantization(df[original_feature])

        days_train = days[train]
        days_val = days[val]
        if len(days_train) < 2:
            print('Skipping this fold since we cannot truncate the last day.')
            continue
        if (lookback_window is not None) and (len(days_train) > lookback_window):
            days_train_valid = days_train[-lookback_window-1:-1]
        else:
            days_train_valid = days_train[:-1]

        X_train, y_train_true = df.loc[(days_train_valid,), :][feature_columns], df.loc[(days_train_valid,), :][return_column]
        X_val, y_val_true = df.loc[(days_val,), :][feature_columns], df.loc[(days_val,), :][return_column]
        if isinstance(training, SupportsPredict):
            y_train_prediction = Series(training.predict(X_train), index=y_train_true.index)
            y_val_prediction = Series(training.predict(X_val), index=y_val_true.index)
        else:
            model = training(X_train, y_train_true)
            y_train_prediction = Series(model.predict(X_train), index=y_train_true.index)
            y_val_prediction = Series(model.predict(X_val), index=y_val_true.index)

        cum_y_val_true = pd.concat([cum_y_val_true, y_val_true])
        cum_y_val_prediction = pd.concat([cum_y_val_prediction, y_val_prediction])

        train_r2 = r2_score(y_train_true, y_train_prediction)
        performance[fold, 'train_r2'] = train_r2
        val_r2 = r2_score(y_val_true, y_val_prediction)
        performance[fold, 'val_r2'] = val_r2
        val_pearson = y_val_true.corr(y_val_prediction)
        performance[fold, 'val_pearson'] = val_pearson
        val_cum_r2 = r2_score(cum_y_val_true, cum_y_val_prediction)
        performance[fold, 'val_cum_r2'] = val_cum_r2
        val_cum_pearson = cum_y_val_true.corr(cum_y_val_prediction)
        performance[fold, 'val_cum_pearson'] = val_cum_pearson

        pbar.set_description(f'Fold {fold}, val_cum_r2={val_cum_r2:.4f}, val_cum_pearson={val_cum_pearson:.4f}')

    return performance


def nan_series_factory(index, name) -> Series:
    data = np.empty(len(index))
    data.fill(np.nan)
    return Series(data, index, dtype=float, name=name)


def nan_dataframe_factory(index, columns) -> DataFrame:
    data = np.empty((len(index), len(columns)))
    data.fill(np.nan)
    return DataFrame(data, index, columns, dtype=float)


def evaluation_for_submission(model: SupportsPredict, feature_columns: Iterable[str], dataset: Dataset, df: DataFrame,
                              qids: QIDS, lookback_window: Union[int, None] = 200,
                              pre_allocate: bool = True) -> Performance:
    """
    Evaluate the given model on the test dataset for submission.
    Assuming no additional features except the fundamental data and extracted market data.

    :param dataset:
    :param df:
    :param qids_path_prefix:
    :return:
    """
    # Assuming that the days start from 1; needs to be checked
    _cum_daily_close = dataset.market.reset_index(['day', 'asset']).loc[N_timeslot].set_index(['day', 'asset'])['close']
    _cum_return_true = dataset.ref_return['return'].rename('ref_return')
    if pre_allocate:
        multi_index_from_day_1 = MultiIndex.from_product([range(1, N_train_days + N_test_days + 1), range(N_asset)],
                                                         names=['day', 'asset'])
        multi_index_from_testing = MultiIndex.from_product(
            [range(N_train_days + 1, N_train_days + N_test_days + 1), range(N_asset)], names=['day', 'asset'])

        cum_daily_close = nan_series_factory(multi_index_from_day_1, 'close')
        cum_daily_close.iloc[:len(_cum_daily_close)] = _cum_daily_close

        cum_return_true = nan_series_factory(multi_index_from_day_1, 'ref_return')
        cum_return_true.iloc[:len(_cum_return_true)] = _cum_return_true

        cum_return_pred = nan_series_factory(multi_index_from_testing, 'pred_return')

        cum_df = nan_dataframe_factory(multi_index_from_day_1, df.columns)
        cum_df.iloc[:len(df), :] = df
    else:
        cum_daily_close = _cum_daily_close.copy()
        cum_return_true = _cum_return_true.copy()
        # We must specify that the index is of a multi-index
        # otherwise the concatenated series contains indices of tuples
        cum_return_pred = Series(dtype=float, name='pred_return',
                                 index=MultiIndex.from_product([[], []], names=['day', 'asset']))
        cum_df = df.copy()

    performance = Performance()
    env = qids.make_env()
    pbar = trange(N_train_days + 1, N_train_days + N_test_days + 1)
    pbar_iter = iter(pbar)

    while not env.is_end():
        current_day = next(pbar_iter)
        before1_day = current_day - 1
        before2_day = current_day - 2

        # Obtain a slice of today's data and append to the full dataframe
        f_current_slice_raw, m_current_slice_raw = env.get_current_market()
        m_current_slice = pre_process_df_with_date_time(m_current_slice_raw)
        f_current_slice = pre_process_df_with_date(f_current_slice_raw)
        m_intraday_slice_df = extract_market_data(m_current_slice)
        current_slice = pd.concat([f_current_slice, m_intraday_slice_df], axis=1)

        current_close = m_current_slice.loc[(current_day, range(N_asset), N_timeslot), 'close'].reset_index('timeslot',
                                                                                                            drop=True)
        if pre_allocate:
            cum_daily_close.iloc[(current_day - 1) * N_asset:current_day * N_asset] = current_close
            cum_df.iloc[(current_day - 1) * N_asset:current_day * N_asset, :] = current_slice
        else:
            cum_daily_close = pd.concat([cum_daily_close, current_close])
            cum_df = pd.concat([cum_df, current_slice])

        ret_n2_true = Series(
            data=(cum_daily_close.loc[(current_day,)].values / cum_daily_close.loc[(before2_day,)].values) - 1,
            index=MultiIndex.from_product([[before2_day], range(N_asset)], names=['day', 'asset']), name='ref_return')
        if pre_allocate:
            cum_return_true.iloc[(before2_day - 1) * N_asset:before2_day * N_asset] = ret_n2_true
        else:
            cum_return_true = pd.concat([cum_return_true, ret_n2_true])

        if isinstance(model, SupportsPredict):
            train_r2 = 0  # since the model does not require re-train

            additional_features, _ = data_quantization(current_slice)
            all_features = pd.concat([current_slice, additional_features], axis=1)

            current_prediction = Series(data=model.predict(all_features.loc[(current_day,), feature_columns]),
                                        index=current_slice.index, name='pred_return')
            assert current_prediction.index.is_monotonic_increasing
            env.input_prediction(current_prediction)
        else:
            # First, train the model on what we already have
            train_start_date = max(1, before1_day - lookback_window) if lookback_window is not None else 1
            sub_cum_df = cum_df.loc[(range(train_start_date, current_day + 1),), :]
            # additional_features, _ = data_quantization(sub_cum_df)
            all_features = sub_cum_df
            # all_features = pd.concat([sub_cum_df, additional_features], axis=1)
            feature_to_last_day = all_features.loc[(range(train_start_date, before1_day),), feature_columns]
            target_to_last_day = cum_return_true.loc[(range(train_start_date, before1_day),)]
            model_obj = model(feature_to_last_day, target_to_last_day)
            train_r2 = r2_score(target_to_last_day, model_obj.predict(feature_to_last_day))

            current_prediction = Series(data=model_obj.predict(all_features.loc[(current_day,), feature_columns]),
                                        index=current_slice.index, name='pred_return')
            assert current_prediction.index.is_monotonic_increasing
            env.input_prediction(current_prediction)

        if pre_allocate:
            cum_return_pred.iloc[
            (current_day - 1 - N_train_days) * N_asset:(current_day - N_train_days) * N_asset] = current_prediction
        else:
            cum_return_pred = pd.concat([cum_return_pred, current_prediction])

        performance[current_day, 'train_r2'] = train_r2
        if before2_day > N_train_days:
            ret_n2_pred = cum_return_pred.loc[(before2_day,)]
            test_r2 = r2_score(ret_n2_true, ret_n2_pred)
            performance[before2_day, 'test_r2'] = test_r2
            test_pearson = ret_n2_true.corr(ret_n2_pred)
            performance[before2_day, 'test_pearson'] = test_pearson

            cum_true = cum_return_true.loc[(range(N_train_days + 1, before1_day),)]
            cum_pred = cum_return_pred.loc[(range(N_train_days + 1, before1_day),)]
            test_cum_r2 = r2_score(cum_true, cum_pred)
            performance[before2_day, 'test_cum_r2'] = test_cum_r2
            test_cum_pearson = cum_true.corr(cum_pred)
            performance[before2_day, 'test_cum_pearson'] = test_cum_pearson

            pbar.set_description(f'Day {current_day}, test cum pearson {test_cum_pearson:.4f}')

    return performance


class Test(TestCase):
    def test_cross_validation(self):
        """
        A minimalist set-up to show how to use the backtest cross-validation suite.
        """
        from sklearn.linear_model import LinearRegression
        from visualization.metric import plot_performance
        from matplotlib import pyplot as plt
        from datatools import data_quantization
        from pipeline import load_mini_dataset

        dataset = load_mini_dataset('../data/parsed_mini', 10)
        quantized_fundamental, _ = data_quantization(dataset.fundamental)
        df = pd.concat([quantized_fundamental, dataset.fundamental, dataset.ref_return], axis=1).dropna()
        quantile_feature = ['turnoverRatio_QUANTILE', 'transactionAmount_QUANTILE', 'pb_QUANTILE', 'ps_QUANTILE',
                            'pe_ttm_QUANTILE', 'pe_QUANTILE', 'pcf_QUANTILE']
        original_feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe_ttm', 'pe', 'pcf']

        def linear_model(X: DataFrame, y: Series):
            reg = LinearRegression().fit(X, y)
            return reg

        performance = cross_validation(linear_model, quantile_feature, df=df, n_splits=9, lookback_window=5)

        plt.figure()
        plot_performance(performance, metrics_selected=['train_r2', 'val_cum_pearson'])
        plt.show()

    def test_evaluation_for_submission(self):
        """
        A minimalist set-up to show how to use the evalution and submission suite.
        """
        from sklearn.linear_model import LinearRegression
        from visualization.metric import plot_performance
        from matplotlib import pyplot as plt
        from pipeline import Dataset

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

        # q_df, _ = data_quantization(dataset.fundamental)
        # full_df = pd.concat([q_df, dataset.ref_return], axis=1).dropna()
        # model = linear_model(full_df[f_quantile_feature], full_df['return'])

        qids = QIDS(path_prefix='../')
        performance = evaluation_for_submission(linear_model, feature, dataset=dataset, df=df, qids=qids,
                                                lookback_window=200)

        plt.figure()
        plot_performance(performance, metrics_selected=['train_r2', 'test_cum_r2', 'test_cum_pearson'])
        plt.show()

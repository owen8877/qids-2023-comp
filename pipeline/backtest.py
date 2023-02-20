from abc import abstractmethod
from typing import Union, Tuple, List, Callable, Iterable, Optional
from typing import runtime_checkable, Protocol  # ERASE_MAGIC
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm, trange

from datatools import extract_market_data, data_quantization, check_dataframe, calculate_market_return
from pipeline import Dataset, N_train_days, N_asset, N_timeslot, N_test_days
from pipeline.parse_raw_df import pre_process_df_with_date_time, pre_process_df_with_date
from qids_lib import QIDS
from visualization.metric import Performance


@runtime_checkable  # ERASE_MAGIC
class ModelLike(Protocol):  # ERASE_MAGIC
    """An ABC with one abstract method `predict`. # ERASE_MAGIC"""
    __slots__ = ()  # ERASE_MAGIC

    @abstractmethod  # ERASE_MAGIC
    def fit_predict(self, X: DataFrame, y: Series) -> Series:  # ERASE_MAGIC
        pass  # ERASE_MAGIC

    @abstractmethod  # ERASE_MAGIC
    def predict(self, X: DataFrame) -> Series:  # ERASE_MAGIC
        pass  # ERASE_MAGIC


Strings = Union[List[str], Tuple[str]]
idx = pd.IndexSlice


def cross_validation(model: ModelLike, feature_columns: Strings, df: DataFrame = None, return_column: str = 'return',
                     train_lookback: Optional[int] = None, per_eval_lookback: int = 1) -> Tuple[Performance, DataFrame]:
    """
    Perform cross validation backtest on the given set of features.

    :param model: a model-like that supports fitting and predicting.
    :param feature_columns:
    :param df: the full dataframe containing all necessary data
    :param return_column:
    :param train_lookback: max number of days of the training feature dataframe, `None` means no truncation.
    :param per_eval_lookback: specifies how many days are need for evaluating the prediction on one validation day.
    :return: performance: a dictionary containing the metric evaluation for each fold.
    :return: cum_y_val_df: a dataframe containing the progressive prediction of y on the validation set and the true values.
    """
    if df is None:
        dataset = Dataset.load(f'{__file__}/../data/parsed')
        df = pd.concat([dataset.fundamental, dataset.ref_return], axis=1).dropna()

    check_dataframe(df, expect_index=['day', 'asset'], expect_feature=feature_columns + [return_column])

    days = df.index.get_level_values('day').unique()
    start_day = df.index.get_level_values('day').min()
    N_days = len(days)
    if train_lookback is not None:
        pbar = trange(per_eval_lookback + train_lookback + start_day, N_days + 1)
    else:
        pbar = trange(2 + start_day, N_days + 1)

    performance = Performance()
    VAL_PRED_LABEL = 'cum_y_val_prediction'
    VAL_TRUE_LABEL = 'cum_y_val_true'
    cum_y_val_df = DataFrame(columns=[VAL_PRED_LABEL, VAL_TRUE_LABEL], dtype=float)

    for val_index in pbar:
        days_train = np.arange(start_day if train_lookback is None else
                               (val_index - per_eval_lookback - train_lookback), val_index - 1)
        days_val = np.arange(val_index + 1 - per_eval_lookback, val_index + 1)

        X_train = df.loc[idx[days_train, :], feature_columns]
        y_train_true = df.loc[idx[days_train[per_eval_lookback - 1:], :], return_column]
        y_train_pred = model.fit_predict(X_train, y_train_true)
        # print(y_train_pred.shape)
        y_train_prediction = Series(y_train_pred, index=y_train_true.index)  # TODO: check shape

        X_val = df.loc[idx[days_val, :], feature_columns]
        y_val_true = df.loc[idx[days_val[per_eval_lookback - 1:], :], return_column]
        y_val_prediction = Series(model.predict(X_val), index=y_val_true.index)

        cum_y_val_df = pd.concat([cum_y_val_df, DataFrame({
            VAL_PRED_LABEL: y_val_prediction,
            VAL_TRUE_LABEL: y_val_true,
        })], axis=0, copy=False)

        train_r2 = r2_score(y_train_true, y_train_prediction)
        performance[val_index, 'train_r2'] = train_r2
        val_r2 = r2_score(y_val_true, y_val_prediction) if len(y_val_true) > 1 else np.nan
        performance[val_index, 'val_r2'] = val_r2
        val_pearson = y_val_true.corr(y_val_prediction) if len(y_val_true) > 1 else np.nan
        performance[val_index, 'val_pearson'] = val_pearson
        val_cum_r2 = r2_score(cum_y_val_df[VAL_TRUE_LABEL], cum_y_val_df[VAL_PRED_LABEL])
        performance[val_index, 'val_cum_r2'] = val_cum_r2
        val_cum_pearson = cum_y_val_df[VAL_TRUE_LABEL].corr(cum_y_val_df[VAL_PRED_LABEL])
        performance[val_index, 'val_cum_pearson'] = val_cum_pearson

        pbar.set_description(
            f'Validation on day {val_index}, train_r2={train_r2:.4f}, val_r2={val_r2:.4f}, val_cum_r2={val_cum_r2:.4f}, val_cum_pearson={val_cum_pearson:.4f}')

    cum_y_val_df.index = MultiIndex.from_tuples(cum_y_val_df.index, names=['day', 'asset'])

    return performance, cum_y_val_df


def nan_series_factory(index, name) -> Series:
    data = np.empty(len(index))
    data.fill(np.nan)
    return Series(data, index, dtype=float, name=name)


def nan_dataframe_factory(index, columns) -> DataFrame:
    data = np.empty((len(index), len(columns)))
    data.fill(np.nan)
    return DataFrame(data, index, columns, dtype=float)


class AugmentationOption:
    def __init__(
            self,
            return_lookback: int = 2,
            market_return: bool = False, market_return_lookback: int = 5,
            quantization: bool = False,
    ):
        self.return_lookback = return_lookback
        self.market_return = market_return
        self.market_return_lookback = market_return_lookback
        self.quantization = quantization


def evaluation_for_submission(model: ModelLike, dataset: Dataset, qids: QIDS, lookback_window: Union[int, None] = 200,
                              per_eval_lookback: int = 1, option: AugmentationOption = None,
                              pre_allocate: bool = True) -> Performance:
    """
    Evaluate the given model on the test dataset for submission.
    Assuming no additional features except the fundamental data and extracted market data.

    :rtype: object
    :param dataset:
    :param qids_path_prefix:
    :return:
    """

    if option is None:
        option = AugmentationOption()

    # Add return data from past few days
    m_df = extract_market_data(dataset.market)
    df = pd.concat([dataset.fundamental, m_df], axis=1)

    return_0 = df['return_0']
    return_list = []
    for i in range(1, option.return_lookback + 1):
        # return_i = return_0.groupby(level=1, group_keys=False).apply(lambda df: df.shift(i).bfill())
        return_i = return_0.shift(N_asset * i).fillna(0)
        return_list.append(return_i.rename(f'return_{i}'))
    df = pd.concat([df, *return_list], axis=1)

    # Add market return
    if option.market_return:
        market_return_0 = calculate_market_return(m_df)
        market_return_list = [market_return_0]
        for i in range(1, option.market_return_lookback + 1):
            market_return_list.append(market_return_0.shift(i).bfill().rename(f'market_return_{i}'))
        market_returns = pd.concat(market_return_list, axis=1)
        df = df.merge(market_returns, left_on='day', right_index=True)

    check_dataframe(df, expect_index=['day', 'asset'])

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
        cum_df = df

    last_market_data = dataset.market.loc[([N_train_days],), :]
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
        m_2intraday_slice_df = extract_market_data(pd.concat([last_market_data, m_current_slice], axis=0))
        m_intraday_slice_df = m_2intraday_slice_df.loc[([current_day],), :]
        current_market_return = calculate_market_return(m_2intraday_slice_df).loc[current_day]

        current_slice = pd.concat([f_current_slice, m_intraday_slice_df], axis=1)
        current_slice['market_return_0'] = current_market_return
        if option.market_return:
            for i in range(1, option.market_return_lookback + 1):
                current_slice[f'market_return_{i}'] = cum_df.loc[(current_day - i, 1), 'market_return_0']
        for i in range(1, option.return_lookback + 1):
            current_slice[f'return_{i}'] = cum_df.loc[idx[[before1_day], :], f'return_{i - 1}'].values
        current_close = m_current_slice.loc[(current_day, range(N_asset), N_timeslot), 'close'].reset_index('timeslot',
                                                                                                            drop=True)
        if pre_allocate:
            cum_daily_close.iloc[(current_day - 1) * N_asset:current_day * N_asset] = current_close
            cum_df.iloc[(current_day - 1) * N_asset:current_day * N_asset, :] = current_slice[cum_df.columns]
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

        last_market_data = m_current_slice

        # First, train the model on what we already have
        days_train = np.arange(1 if lookback_window is None else (current_day - per_eval_lookback - lookback_window),
                               before1_day)

        X_train = cum_df.loc[idx[days_train, :], :]
        y_train_true = cum_return_true.loc[idx[days_train[per_eval_lookback - 1:], :]]
        y_train_pred = model.fit_predict(X_train, y_train_true)
        y_train_prediction = Series(y_train_pred, index=y_train_true.index)  # TODO: check shape
        train_r2 = r2_score(y_train_true, y_train_prediction)

        X_eval = cum_df.loc[idx[(current_day + 1 - per_eval_lookback): current_day, :], :]
        y_eval_prediction = Series(model.predict(X_eval), index=X_eval.index[-N_asset:], name='pred_return')

        assert y_eval_prediction.index.is_monotonic_increasing
        env.input_prediction(y_eval_prediction)

        if pre_allocate:
            cum_return_pred.iloc[
            (current_day - 1 - N_train_days) * N_asset:(current_day - N_train_days) * N_asset] = y_eval_prediction
        else:
            cum_return_pred = pd.concat([cum_return_pred, y_eval_prediction])

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
        import matplotlib as mpl

        dataset = load_mini_dataset('data/parsed_mini', 10, path_prefix='..')
        quantized_fundamental, _ = data_quantization(dataset.fundamental)
        df = pd.concat([quantized_fundamental, dataset.fundamental, dataset.ref_return], axis=1).dropna()
        quantile_feature = ['turnoverRatio_QUANTILE', 'transactionAmount_QUANTILE', 'pb_QUANTILE', 'ps_QUANTILE',
                            'pe_ttm_QUANTILE', 'pe_QUANTILE', 'pcf_QUANTILE']
        original_feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe_ttm', 'pe', 'pcf']

        class SimpleLinearModel(ModelLike):
            def __init__(self):
                self.reg = LinearRegression()

            def fit_predict(self, X: DataFrame, y: Series):
                print('fit:')
                print(X.index.get_level_values(0).unique())
                self.reg.fit(X, y)
                return self.reg.predict(X)

            def predict(self, X: DataFrame):
                print('predict:')
                print(X.index.get_level_values(0).unique())
                return self.reg.predict(X)

        simple_linear_model = SimpleLinearModel()

        performance, cum_y_val_df = cross_validation(simple_linear_model, quantile_feature, df=df, train_lookback=5,
                                                     per_eval_lookback=1)
        performance, cum_y_val_df = cross_validation(simple_linear_model, quantile_feature, df=df, train_lookback=5,
                                                     per_eval_lookback=3)

        mpl.use('TkAgg')
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
        from pipeline import Dataset, load_mini_dataset

        dataset = Dataset.load('../data/parsed')
        # dataset = load_mini_dataset(path_prefix='..')
        df = dataset.fundamental
        f_quantile_feature = ['turnoverRatio_QUANTILE', 'transactionAmount_QUANTILE', 'pb_QUANTILE', 'ps_QUANTILE',
                              'pe_ttm_QUANTILE', 'pe_QUANTILE', 'pcf_QUANTILE']
        m_quantile_feature = ['avg_price_QUANTILE', 'volatility_QUANTILE', 'mean_volume_QUANTILE']
        feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe_ttm', 'pe', 'pcf', 'avg_price', 'volatility',
                   'mean_volume']

        class SimpleLinearModel:
            def __init__(self, features):
                self.reg = LinearRegression(fit_intercept=True)
                self.features = features

            def fit_predict(self, X, y):
                self.reg.fit(X, y)
                y_pred = self.reg.predict(X)
                return y_pred

            def predict(self, X):
                return self.reg.predict(X)

        qids = QIDS(path_prefix='../')
        # qids = None
        model = SimpleLinearModel(feature)
        performance = evaluation_for_submission(model, dataset=dataset, qids=qids, lookback_window=200,
                                                option=AugmentationOption(market_return=True))

        plt.figure()
        plot_performance(performance, metrics_selected=['train_r2', 'test_cum_r2', 'test_cum_pearson'])
        plt.show()

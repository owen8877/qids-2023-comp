from abc import abstractmethod
from typing import Union, Tuple, List, Optional
from typing import runtime_checkable, Protocol  # ERASE_MAGIC
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas import Series, DataFrame, MultiIndex
from sklearn.metrics import r2_score
from tqdm.auto import trange
from xarray import Dataset, DataArray
import xarray as xr

import pipeline
from datatools import extract_market_data, check_dataframe, calculate_market_return
from pipeline import N_train_days, N_asset, N_timeslot, N_test_days
from pipeline.parse_raw_df import pre_process_df
from qids_lib import QIDS
from visualization.metric import Performance


@runtime_checkable  # ERASE_MAGIC
class ModelLike(Protocol):  # ERASE_MAGIC
    """An ABC with one abstract method `predict`. # ERASE_MAGIC"""
    __slots__ = ()  # ERASE_MAGIC

    @abstractmethod  # ERASE_MAGIC
    def fit_predict(self, X: Dataset, y: DataArray) -> DataArray:  # ERASE_MAGIC
        pass  # ERASE_MAGIC

    @abstractmethod  # ERASE_MAGIC
    def predict(self, X: Dataset) -> DataArray:  # ERASE_MAGIC
        pass  # ERASE_MAGIC


Strings = Union[List[str], Tuple[str]]
idx = pd.IndexSlice


def cross_validation(model: ModelLike, feature_columns: Strings, ds: Dataset = None, return_column: str = 'return',
                     train_lookback: Optional[int] = None, per_eval_lookback: int = 1) -> Tuple[Performance, Dataset]:
    """
    Perform cross validation backtest on the given set of features.

    :param model: a model-like that supports fitting and predicting.
    :param feature_columns:
    :param ds: the full dataframe containing all necessary data
    :param return_column:
    :param train_lookback: max number of days of the training feature dataframe, `None` means no truncation.
    :param per_eval_lookback: specifies how many days are need for evaluating the prediction on one validation day.
    :return: performance: a dictionary containing the metric evaluation for each fold.
    :return: cum_y_val_df: a dataframe containing the progressive prediction of y on the validation set and the true values.
    """
    if ds is None:
        # TODO: refactor loading code
        dataset = pipeline.Dataset.load(f'{__file__}/../data/parsed')
        ds = Dataset.from_dataframe(pd.concat([dataset.fundamental, dataset.ref_return], axis=1).dropna())

    # TODO: refactor check dataset code
    # check_dataframe(ds, expect_index=['day', 'asset'], expect_feature=feature_columns + [return_column])
    assert {'day', 'asset'}.issubset(set(ds.dims.keys()))
    assert set(feature_columns + [return_column]).issubset(set(ds.variables))

    start_day = ds.day.min().item()
    end_day = ds.day.max().item()
    val_start_day = (2 + start_day) if train_lookback is None else (per_eval_lookback + train_lookback + start_day)
    pbar = trange(val_start_day, end_day + 1)

    performance = Performance()
    coords = ds.sel(day=slice(val_start_day, end_day)).coords
    cum_y_val_prediction = DataArray(np.nan, coords=coords)
    cum_y_val_true = DataArray(np.nan, coords=coords)

    for val_index in pbar:
        days_train = np.arange(start_day if train_lookback is None else
                               (val_index - per_eval_lookback - train_lookback), val_index - 1)
        days_val = np.arange(val_index + 1 - per_eval_lookback, val_index + 1)

        X_train = ds[feature_columns].sel(day=days_train)
        y_train_true = ds[return_column].sel(day=days_train[per_eval_lookback - 1:])
        y_train_pred = model.fit_predict(X_train, y_train_true)
        y_train_prediction = DataArray(data=y_train_pred, coords=y_train_true.coords)  # TODO: check shape

        X_val = ds[feature_columns].sel(day=days_val)
        y_val_true = ds[return_column].sel(day=days_val[per_eval_lookback - 1:])
        y_val_prediction = DataArray(data=model.predict(X_val), coords=y_val_true.coords)

        cum_y_val_true.loc[dict(day=val_index)] = y_val_true.sel(day=val_index)
        cum_y_val_prediction.loc[dict(day=val_index)] = y_val_prediction.sel(day=val_index)

        train_r2 = r2_score(y_train_true.to_series(), y_train_prediction.to_series())
        performance[val_index, 'train_r2'] = train_r2
        val_r2 = r2_score(y_val_true.to_series(), y_val_prediction.to_series()) if len(
            y_val_true.to_series()) > 1 else np.nan
        performance[val_index, 'val_r2'] = val_r2
        val_pearson = y_val_true.to_series().corr(y_val_prediction.to_series()) if len(
            y_val_true.to_series()) > 1 else np.nan
        performance[val_index, 'val_pearson'] = val_pearson

        y_val_true_so_far = cum_y_val_true.sel(day=slice(start_day, val_index)).to_series()
        y_val_pred_so_far = cum_y_val_prediction.sel(day=slice(start_day, val_index)).to_series()
        val_cum_r2 = r2_score(y_val_true_so_far, y_val_pred_so_far)
        performance[val_index, 'val_cum_r2'] = val_cum_r2
        val_cum_pearson = y_val_true_so_far.corr(y_val_pred_so_far)
        performance[val_index, 'val_cum_pearson'] = val_cum_pearson

        # raise

        pbar.set_description(
            f'Validation on day {val_index}, train_r2={train_r2:.4f}, val_r2={val_r2:.4f}, val_cum_r2={val_cum_r2:.4f}, val_cum_pearson={val_cum_pearson:.4f}')

    cum_y_val_df = Dataset({k.name: k for k in (cum_y_val_true, cum_y_val_prediction)})

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


def evaluation_for_submission(model: ModelLike, given_ds: Dataset, qids: QIDS, lookback_window: Union[int, None] = 200,
                              per_eval_lookback: int = 1, option: AugmentationOption = None) -> Performance:
    """
    Evaluate the given model on the test dataset for submission.
    Assuming no additional features except the fundamental data and extracted market data.
    """
    N_total = N_train_days + N_test_days
    more_ds = Dataset(data_vars={k: np.nan for k in given_ds.data_vars},
                      coords=dict(day=range(N_train_days + 1, N_total + 1), asset=range(N_asset),
                                  timeslot=range(1, N_timeslot + 1)))
    ds: Dataset = xr.combine_by_coords([given_ds, more_ds])

    if option is None:
        option = AugmentationOption()

    # Add return data from past few days
    return_0 = ds['return_0']
    for i in range(1, option.return_lookback + 1):
        return_i = return_0.shift(day=i, fill_value=0)
        ds[f'return_{i}'] = return_i

    # Add market return
    if option.market_return:
        market_return_0 = calculate_market_return(ds['return_0'])
        ds['market_return_0'] = market_return_0
        for i in range(1, option.market_return_lookback + 1):
            ds[f'market_return_{i}'] = market_return_0.shift(day=i, fill_value=0)

    # Assuming that the days start from 1; needs to be checked
    ds['return_pred'] = DataArray(data=np.nan, coords=dict(day=range(1, N_total + 1), asset=range(N_asset)),
                                  dims=['day', 'asset'])
    all_features_but_return = list(set(ds.data_vars.keys()) - {'return', 'return_pred'})

    performance = Performance()
    env = qids.make_env()
    pbar = trange(N_train_days + 1, N_train_days + N_test_days + 1)
    pbar_iter = iter(pbar)

    while not env.is_end():
        current_day = next(pbar_iter)
        before1_day = current_day - 1
        before2_day = current_day - 2

        # Obtain a slice of today's data and update to the full dataset
        new_fundamental_raw, new_market_raw = env.get_current_market()
        new_ds = pre_process_df(new_fundamental_raw, new_market_raw)
        for col in new_ds.data_vars:
            ds[col].loc[dict(day=current_day)] = new_ds[col].sel(day=current_day)

        new_market_brief = extract_market_data(
            ds[['open', 'close', 'low', 'high', 'volume', 'money']].sel(day=[before1_day, current_day])
        ).sel(day=[current_day])
        for col in new_market_brief:
            ds[col].loc[dict(day=current_day)] = new_market_brief[col].sel(day=current_day)

        return_forecast_true = ds['close_0'].sel(day=current_day) / ds['close_0'].sel(day=before2_day) - 1
        ds['return'].loc[dict(day=before2_day)] = return_forecast_true

        # Data augmentation
        if option.market_return:
            current_market_return = calculate_market_return(new_market_brief['return_0'])
            ds[current_market_return.name].loc[dict(day=current_day)] = current_market_return.sel(day=current_day)
            for i in range(1, option.market_return_lookback + 1):
                ds[f'market_return_{i}'].loc[dict(day=current_day)] = ds['market_return_0'].sel(day=current_day - i)
        for i in range(1, option.return_lookback + 1):
            ds[f'return_{i}'].loc[dict(day=current_day)] = ds[f'return_{i - 1}'].sel(day=before1_day)

        # Train the model on what we already have
        days_train = range(1 if lookback_window is None else (current_day - per_eval_lookback - lookback_window),
                           before1_day)

        X_train = ds[all_features_but_return].sel(day=days_train)
        assert not X_train.isnull().any().to_array().any().item()
        y_train_true = ds['return'].sel(day=days_train[per_eval_lookback - 1:])
        y_train_pred = model.fit_predict(X_train, y_train_true)
        train_r2 = r2_score(y_train_true.to_series(), y_train_pred.to_series())

        X_eval = ds[all_features_but_return].sel(day=slice(current_day + 1 - per_eval_lookback, current_day))
        assert not X_eval.isnull().any().to_array().any().item()
        y_eval_prediction = model.predict(X_eval).rename('return_pred')
        assert set(y_eval_prediction.dims) == {'day', 'asset'}
        assert y_eval_prediction.asset.to_series().is_monotonic_increasing
        env.input_prediction(y_eval_prediction.transpose('day', 'asset').to_series())
        ds['return_pred'].loc[dict(day=current_day)] = y_eval_prediction.isel(
            day=0)  # TODO: why day information is lost in rolling_exp?

        performance[current_day, 'train_r2'] = train_r2
        if before2_day > N_train_days:
            return_forecast_pred = ds['return_pred'].sel(day=before2_day)
            # print(return_forecast_true.to_series())
            # print(return_forecast_pred.to_series())
            test_r2 = r2_score(return_forecast_true.to_series(), return_forecast_pred.to_series())
            performance[before2_day, 'test_r2'] = test_r2
            test_pearson = return_forecast_true.to_series().corr(return_forecast_pred.to_series())
            performance[before2_day, 'test_pearson'] = test_pearson

            cum_return_true = ds['return'].sel(day=slice(N_train_days + 1, before2_day))
            cum_return_pred = ds['return_pred'].sel(day=slice(N_train_days + 1, before2_day))
            test_cum_r2 = r2_score(cum_return_true.to_series(), cum_return_pred.to_series())
            performance[before2_day, 'test_cum_r2'] = test_cum_r2
            test_cum_pearson = cum_return_true.to_series().corr(cum_return_pred.to_series())
            performance[before2_day, 'test_cum_pearson'] = test_cum_pearson

            pbar.set_description(f'Day {current_day}, train r2={train_r2:.4f}, test cum r2={test_cum_r2:.4f}, '
                                 f'test cum pearson {test_cum_pearson:.4f}')

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

        performance, cum_y_val_df = cross_validation(simple_linear_model, quantile_feature, ds=df, train_lookback=5,
                                                     per_eval_lookback=1)
        performance, cum_y_val_df = cross_validation(simple_linear_model, quantile_feature, ds=df, train_lookback=5,
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
        from pipeline import Dataset

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

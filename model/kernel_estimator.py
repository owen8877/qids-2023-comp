from typing import Callable

import numpy as np
from pandas import DataFrame, Series
from xarray import Dataset, DataArray


def gaussian_kernel(t):
    return np.exp(-t ** 2)


def cauchy_kernel(t):
    return np.exp(-np.abs(t))


class ConditionalKernelEstimator:
    def __init__(self, kernel: Callable, bandwidth: float = 1):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def fit_predict(self, X: Dataset, y: DataArray) -> DataArray:
        self.X_ref = X
        self.y_ref = y
        return self.predict(X)

    def predict(self, X: Dataset) -> DataArray:
        X_ref_pd: DataFrame = self.X_ref.to_dataframe()
        X_query_pd: DataFrame = X.to_dataframe()[X_ref_pd.columns]
        assert X_ref_pd.index.names == X_query_pd.index.names

        X_ref_np = X_ref_pd.values[np.newaxis, :, :]
        y_ref_np = self.y_ref.to_series().values[np.newaxis, :]
        X_query_np = X_query_pd.values[:, np.newaxis, :]

        weight = self.kernel((X_ref_np - X_query_np) / self.bandwidth).prod(axis=2) + 1e-9
        y_query_np = (y_ref_np * weight).sum(axis=1) / weight.sum(axis=1)
        y_query_pd = Series(data=y_query_np, index=X_query_pd.index)
        return DataArray.from_series(y_query_pd)

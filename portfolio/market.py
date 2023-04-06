from unittest import TestCase

import numpy as np
from xarray import Dataset, DataArray
import xarray as xr

from portfolio import Portfolio

import pipeline.backtest_trading as bt


class MarketPortfolio(Portfolio):
    def __init__(self):
        super().__init__(1, 'market')

    def initialize(self, X: Dataset, y: DataArray):
        pass

    def train(self, X: Dataset, y: DataArray):
        pass

    def construct(self, X: Dataset) -> DataArray:
        return np.ones(54) / 54


class Test(TestCase):
    def test_market_stat(self):
        p = MarketPortfolio()
        base_ds = xr.open_dataset('../data/nc_2round/base.nc')
        fundamental_v0_ds = xr.open_dataset('../data/nc_2round/fundamental_v0.nc')
        market_ds = xr.open_dataset('../data/nc_2round/market_brief.nc')
        ds = xr.merge([base_ds, fundamental_v0_ds, market_ds])
        stat = bt.cross_validation(p, ['close_0'], ds, lookback_window=p.lookback_window)
        p.dump_stat(stat, need_timestamp=False)

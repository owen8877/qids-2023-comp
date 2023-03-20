from abc import abstractmethod
from typing import runtime_checkable, Protocol, Optional, List, Union, Tuple
from unittest import TestCase

import numpy as np
from tqdm.auto import trange
from xarray import Dataset, DataArray


@runtime_checkable
class Portfolio(Protocol):
    __slots__ = ()

    @abstractmethod
    def initialize(self, X: Dataset, y: DataArray):
        pass

    @abstractmethod
    def train(self, X: Dataset, y: DataArray):
        pass

    @abstractmethod
    def construct(self, X: Dataset) -> DataArray:
        pass


Strings = Union[Tuple[str], List[str]]


def nan_factory(**dim_args):
    dims = list(dim_args.keys())
    arr = np.empty([len(dim_args[dim]) for dim in dims])
    arr.fill(np.nan)
    return dims, arr


def cross_validation(
        portfolio: Portfolio, features: Strings, ds: Dataset, lookback_window: Optional[int] = 16,
        need_full_lookback: bool = False,
        open_position_rate: float = 4e-4, close_position_rate: float = 2e-3,
) -> Dataset:
    """

    :param portfolio: assumed initialized but not trained on time span specified by `ds`
    :param features:
    :param ds:
    :param lookback_window:
    :param need_full_lookback
    :return: dataset that contains trading performance
    """
    start_day: int = ds.day.min().item()
    end_day: int = ds.day.max().item()

    transactions = DataArray(np.nan, dims=['day', 'asset'],
                             coords={'asset': ds.asset.to_numpy(), 'day': np.arange(start_day - 2, end_day + 1)})
    transactions.loc[dict(day=range(start_day - 2, start_day))] = 0
    stat = Dataset(data_vars={k: nan_factory(asset=ds.asset, day=ds.day) for k in ['holding_return', 'open_fee', 'close_fee']},
                   coords={'asset': ds.asset, 'day': ds.day})

    pbar = trange(start_day, end_day + 1)
    for current_day in pbar:
        __d = dict(day=current_day)

        # Clearing stage
        transaction_1 = transactions.sel(day=current_day - 1)
        transaction_2 = transactions.sel(day=current_day - 2)

        stat.holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * (transaction_1 + transaction_2) / 2

        # Transaction stage
        if lookback_window is None:
            chunk_start_day = start_day
        else:
            chunk_start_day = current_day - lookback_window + 1

        def whether_to_construct():
            if need_full_lookback and (chunk_start_day < start_day):
                pbar.set_description(f'Skipped since the lookback chunk has not yet reached full length.')
                return False
            return current_day <= end_day - 2

        to_construct = whether_to_construct()

        if to_construct:
            sub_ds = ds[features].sel(day=slice(chunk_start_day, current_day))
            tr = portfolio.construct(sub_ds)
        else:
            tr = 0
        # TODO: sanity check
        transactions.loc[dict(day=[current_day])] = tr
        transaction_0 = transactions.sel(day=current_day)

        # Post-transaction fee calculation
        marginal_transaction = transaction_0 - transaction_2
        stat.open_fee.loc[__d] = open_position_rate * marginal_transaction.clip(0) / 2
        stat.close_fee.loc[__d] = (-close_position_rate) * marginal_transaction.clip(None, 0) / 2

        # Train the portfolio
        if to_construct:
            portfolio.train(sub_ds, ds['return'].sel(day=slice(chunk_start_day, current_day)))

    return stat


class Test(TestCase):
    def test_compare_with_example(self):
        import xarray as xr
        few_ds = Dataset(data_vars={'apple': (['asset', 'day'], np.zeros((4, 5))),
                                    'return': (['asset', 'day'], np.zeros((4, 5)))},
                         coords={'asset': [1, 2, 3, 4], 'day': [1701, 1702, 1703, 1704, 1705]})
        return0_ds = DataArray(
            [[0, 0, 0, 0], [0.01, 0.04, -0.03, 0.1], [0.02, 0, -0.02, 0.1], [0, 0.08, -0.1, 0.1],
             [-0.01, 0.06, 0, -0.1]], coords={'asset': few_ds.asset, 'day': [1701, 1702, 1703, 1704, 1705]},
            dims=['day', 'asset'])
        ds = xr.merge([few_ds, return0_ds.to_dataset(name='return_0')])

        class Portfolio:
            def train(self, X: Dataset, y: DataArray):
                pass

            def construct(self, X: Dataset):
                current_day = X.day.item()
                asset_dict = {'asset': X.asset}
                if current_day == 1701:
                    return DataArray([0.5, 0.5, 0, 0], coords=asset_dict)
                elif current_day == 1702:
                    return DataArray([0.5, 0.25, 0.25, 0], coords=asset_dict)
                elif current_day == 1703:
                    return DataArray([0, 0.25, 0, 0.5], coords=asset_dict)
                else:
                    raise ValueError

        portfolio = Portfolio()
        stat = cross_validation(portfolio, ['apple'], ds, lookback_window=1)

        s = stat.sum(dim='asset')
        print(s)
        pnl = s.holding_return - s.open_fee.shift(day=1) - s.close_fee
        print(pnl)
        self.assertTrue(np.isclose(pnl.sel(day=1702), (
                0.5 * (0.01 * 0.5 + 0.04 * 0.5)
                - 4e-4 * 0.5 * (0.5 + 0.5)
        )))
        self.assertTrue(np.isclose(pnl.sel(day=1703), (
                0.5 * (0.02 * 0.5 + 0 * 0.5)
                + 0.5 * (0.02 * 0.5 + 0 * 0.25 - 0.02 * 0.25)
                - 2e-3 * 0.5 * (0.5 + 0.5 - 0.25)
                - 4e-4 * 0.5 * (0.5 + 0.25 + 0.25)
        )))
        self.assertTrue(np.isclose(pnl.sel(day=1704), (
                0.5 * (0 * 0.5 + 0.08 * 0.25 - 0.1 * 0.25)
                + 0.5 * (0.08 * 0.25 + 0.1 * 0.5)
                - 2e-3 * 0.5 * (0.5 + 0.25 + 0.25)
                - 4e-4 * 0.5 * (0.25 - 0.25 + 0.5)
        )))
        self.assertTrue(np.isclose(pnl.sel(day=1705), (
                0.5 * (0.06 * 0.25 - 0.1 * 0.5)
                - 2e-3 * 0.5 * (0.25 + 0.5)
        )))

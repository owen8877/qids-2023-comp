from typing import Optional, List, Union, Tuple
from unittest import TestCase

import numpy as np
import pandas as pd
from tqdm.auto import trange
from xarray import Dataset, DataArray

from portfolio import Portfolio

Strings = Union[Tuple[str], List[str]]


def nan_factory(**dim_args):
    dims = list(dim_args.keys())
    arr = np.empty([len(dim_args[dim]) for dim in dims])
    arr.fill(np.nan)
    return dims, arr


def normalize_and_check_transaction(tr):
    if isinstance(tr, float) or isinstance(tr, int):
        if tr < 0 or 0 < tr < 0.01 or tr > 1 / 54:
            raise ValueError(f'The uniform transaction amount {tr:.4f} if either too small or too large!')
        return tr
    if isinstance(tr, DataArray):
        tr = tr.to_numpy()
    elif isinstance(tr, pd.Series):
        tr = tr.to_numpy()
    elif isinstance(tr, list):
        tr = np.array(tr)
    if not isinstance(tr, np.ndarray):
        raise ValueError(f'Type {type(tr)} not recognized!')
    tr = tr.squeeze()

    assert len(tr) == 54, 'Length shall be 54'
    assert np.sum(tr) <= 1 + 1e-10, 'Sum shall not exceed 1'
    assert np.all(tr >= 0), 'All terms shall be non-negative'
    assert np.all(tr[tr > 0] >= 0.01), 'Some positive entries are smaller than 0.01'
    return tr


def cross_validation(
        portfolio: Portfolio, features: Strings, ds: Dataset, lookback_window: Optional[int] = 16,
        need_full_lookback: bool = False,
        open_position_rate: float = 4e-4, close_position_rate: float = 2e-3,
        start_day: Optional[int] = None,
        disable_portfolio_initialization: bool = False,
) -> Dataset:
    """

    :param portfolio: assumed initialized but not trained on time span specified by `ds`
    :param features:
    :param ds:
    :param lookback_window:
    :param need_full_lookback
    :return: dataset that contains trading performance
    """
    ds_min_day = ds.day.min().item()
    start_day: int = ds_min_day if start_day is None else start_day
    end_day: int = ds.day.max().item()

    transactions = DataArray(np.nan, dims=['day', 'asset'],
                             coords={'asset': ds.asset.to_numpy(), 'day': np.arange(start_day - 2, end_day + 1)})
    transactions.loc[dict(day=range(start_day - 2, start_day))] = 0
    stat = Dataset(
        data_vars={k: nan_factory(asset=ds.asset, day=ds.day) for k in [
            'holding_return', 'open_fee', 'close_fee',
            'odd_holding_return', 'odd_open_fee', 'odd_close_fee',
            'even_holding_return', 'even_open_fee', 'even_close_fee',
        ]},
        coords={'asset': ds.asset, 'day': ds.day})

    pbar = trange(start_day, end_day + 1)
    odd_cum_log_return, even_cum_log_return = 0.0, 0.0
    if not disable_portfolio_initialization:
        init_slice = slice(start_day - lookback_window - 2, start_day - 2)
        portfolio.initialize(ds[features].sel(day=init_slice), ds['return'].sel(day=init_slice))
    for current_day in pbar:
        __d = dict(day=current_day)

        # Clearing stage
        transaction_1 = transactions.sel(day=current_day - 1)
        transaction_2 = transactions.sel(day=current_day - 2)

        stat.holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * (transaction_1 + transaction_2) / 2
        adjustment = 1 if current_day == start_day else (1 + ds['return_0'].sel(day=current_day - 1))
        if current_day % 2 == 0:
            # tr_1 is odd and tr_2 is even
            stat.odd_holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * transaction_1
            stat.even_holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * transaction_2 * adjustment
        else:
            # tr_2 is odd and tr_1 is even
            stat.odd_holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * transaction_2 * adjustment
            stat.even_holding_return.loc[__d] = ds['return_0'].sel(day=current_day) * transaction_1

        # Transaction stage
        if lookback_window is None:
            chunk_start_day = start_day
        else:
            chunk_start_day = current_day - lookback_window + 1

        def whether_to_construct():
            if need_full_lookback and (chunk_start_day < ds_min_day):
                pbar.set_description(f'Skipped since the lookback chunk has not yet reached full length.')
                return False
            return current_day <= end_day - 2

        to_construct = whether_to_construct()

        if to_construct:
            train_slice = slice(chunk_start_day - 2, current_day - 2)
            portfolio.train(ds[features].sel(day=train_slice), ds['return'].sel(day=train_slice))
            tr = portfolio.construct(ds[features].sel(day=slice(chunk_start_day, current_day)))
        else:
            tr = 0.0
        tr = normalize_and_check_transaction(tr)
        transactions.loc[dict(day=[current_day])] = tr
        transaction_0 = transactions.sel(day=current_day)

        # Post-transaction fee calculation
        marginal_transaction = transaction_0 - transaction_2
        stat.open_fee.loc[__d] = open_position_rate * marginal_transaction.clip(0) / 2
        stat.close_fee.loc[__d] = (-close_position_rate) * marginal_transaction.clip(None, 0) / 2
        if current_day % 2 == 0:
            # handles an even account
            stat.even_open_fee.loc[__d] = open_position_rate * marginal_transaction.clip(0)
            stat.even_close_fee.loc[__d] = (-close_position_rate) * marginal_transaction.clip(None, 0)
            stat.odd_open_fee.loc[__d] = 0
            stat.odd_close_fee.loc[__d] = 0
        else:
            # handles an odd account
            stat.odd_open_fee.loc[__d] = open_position_rate * marginal_transaction.clip(0)
            stat.odd_close_fee.loc[__d] = (-close_position_rate) * marginal_transaction.clip(None, 0)
            stat.even_open_fee.loc[__d] = 0
            stat.even_close_fee.loc[__d] = 0

        # PnL display; may not be perfect since open fee is calculated differently
        if current_day % 2 == 0:
            odd_cum_log_return += np.log(
                1 + stat.odd_holding_return.sel(day=current_day).sum(dim='asset')
            )
            even_cum_log_return += np.log(
                1 + stat.even_holding_return.sel(day=current_day).sum(dim='asset')
                - stat.even_open_fee.sel(day=current_day).sum(dim='asset')
                - stat.even_close_fee.sel(day=current_day).sum(dim='asset')
            )
        else:
            even_cum_log_return += np.log(
                1 + stat.even_holding_return.sel(day=current_day).sum(dim='asset')
            )
            odd_cum_log_return += np.log(
                1 + stat.odd_holding_return.sel(day=current_day).sum(dim='asset')
                - stat.odd_open_fee.sel(day=current_day).sum(dim='asset')
                - stat.odd_close_fee.sel(day=current_day).sum(dim='asset')
            )

        money = (np.exp(odd_cum_log_return) + np.exp(even_cum_log_return)) / 2
        pbar.set_description(f'Total PnL={np.log(money):.4f}')

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

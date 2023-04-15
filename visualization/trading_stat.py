from typing import Tuple, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from xarray import DataArray, Dataset

from pipeline.backtest_trading import Strings, cross_validation
from portfolio import Portfolio


class BackTestPeriod:
    periods = (80, 200), (640, 1000), (1030, 1300)


def _analyze_stat_in_period(stat: Dataset, market_stat: Dataset, period: Tuple[int, int], axs: Iterable[Axes],
                            odd_even: bool):
    ax_pnl, ax_scatter = axs
    start, end = period

    def calculate_return(stat: Dataset, prefix: str) -> DataArray:
        return (stat[f'{prefix}holding_return'] - stat[f'{prefix}open_fee'].shift(day=1) - stat[
            f'{prefix}close_fee']).fillna(0).sum(dim='asset')

    def calculate_pnl(stat: Dataset):
        r = calculate_return(stat, '')
        odd_r = calculate_return(stat, 'odd_')
        even_r = calculate_return(stat, 'even_')
        log_r = np.log(r + 1)

        log_pnl = log_r.cumsum()
        odd_log_pnl = np.log(odd_r + 1).cumsum()
        even_log_pnl = np.log(even_r + 1).cumsum()
        odd_even_pnl = (np.exp(odd_log_pnl) + np.exp(even_log_pnl)) / 2 - 1
        odd_even_log_pnl = np.log(odd_even_pnl + 1)

        start_day = odd_even_log_pnl.day.min().item()
        odd_even_log_r = xr.concat([xr.DataArray([odd_even_log_pnl.sel(day=start_day)], coords={'day': [start_day]}),
                                    odd_even_log_pnl.diff('day')], dim='day')

        return log_r, log_pnl, odd_even_log_r, odd_even_log_pnl

    log_r, log_pnl, odd_even_log_r, odd_even_log_pnl = calculate_pnl(stat.sel(day=slice(start, end)))
    m_log_r, m_log_pnl, m_odd_even_log_r, m_odd_even_log_pnl = calculate_pnl(market_stat.sel(day=slice(start, end)))

    ax_pnl.axhline(y=0, ls=':', alpha=0.2, c='k')
    if odd_even:
        odd_even_log_pnl.plot(ax=ax_pnl, label='*')
        m_odd_even_log_r.plot(ax=ax_pnl, ls='--', label='market', alpha=0.4)
    else:
        log_pnl.plot(ax=ax_pnl, label='*')
        m_log_pnl.plot(ax=ax_pnl, ls='--', label='market', alpha=0.4)
    ax_pnl.set(xlabel='day', ylabel='log pnl')
    ax_pnl.legend(loc='best')

    ax_scatter.axvline(x=0, ls=':', alpha=0.2, c='k')
    ax_scatter.axhline(y=0, ls=':', alpha=0.2, c='k')
    if odd_even:
        ax_scatter.scatter(m_odd_even_log_r, odd_even_log_r, s=5)
    else:
        ax_scatter.scatter(m_log_r, log_r, s=5)
    ax_scatter.set(xlabel='market log return', ylabel='portfolio log return', xlim=[-0.1, 0.1], ylim=[-0.1, 0.1])

    print('Summary of scores:')
    _r = odd_even_log_r if odd_even else log_r
    _pnl = odd_even_log_pnl if odd_even else log_pnl
    apr = _pnl.isel(day=-1) / (end - start) * 252
    sigma = np.std(_r) * np.sqrt(252)
    print(f'1. APR: {apr * 100:.4f}%')
    print(f'2. Sharpe: {apr / sigma:.4f}')
    running_max = np.maximum.accumulate(_r.to_numpy())
    print(f'3. Maximum drawdown: {np.max(running_max - _r.to_numpy()):.4f}')
    print(f'4. Median number of traded instruments: {(stat.holding_return != 0).sum(dim="asset").median():.1f}')


def analyze_stat(stat: Dataset, path_prefix='../..', odd_even: bool = False):
    try:
        market_portfolio_stat = xr.open_dataset(f'{path_prefix}/portfolio/stat/market.nc')
    except FileNotFoundError:
        print('You need to generate the market portfolio stat first! Go and run task `Export market strategy`.')
        raise FileNotFoundError

    stat_start, stat_end = stat.day.min().item(), stat.day.max().item()
    periods_to_plot = []
    for period in BackTestPeriod.periods:
        period_start, period_end = period
        start, end = max(stat_start, period_start), min(stat_end, period_end)
        if end - start < 10:
            # skip this period since it is too short for analysis
            continue

        periods_to_plot.append((start, end))

    n_periods = len(periods_to_plot)
    if n_periods < 1:
        raise ValueError('Not enough periods for analysis; have you backtested on a bear market?')

    fig, axs = plt.subplots(n_periods, 2, figsize=(15, 5 * n_periods), squeeze=False)
    for ax_row, period in zip(axs, periods_to_plot):
        _analyze_stat_in_period(stat, market_portfolio_stat, period, ax_row, odd_even)


def bear_market_suite(portfolio: Portfolio, features: Strings, ds: Dataset, lookback_window: Optional[int] = 16,
                      path_prefix: str = '../..', odd_even: bool = False, **kwargs):
    for period in BackTestPeriod.periods:
        start, end = period
        if lookback_window is None:
            chunk_start = 1
        else:
            chunk_start = start - lookback_window - 5
        _ds = ds.sel(day=slice(chunk_start, end))
        if len(_ds.day) <= 100:
            print(f'Period {period} not available in ds, skipping...')
            continue
        print(f'Now testing period {period}:')
        stat = cross_validation(portfolio, features, _ds, start_day=start,
                                lookback_window=lookback_window, **kwargs)
        analyze_stat(stat, path_prefix=path_prefix, odd_even=odd_even)

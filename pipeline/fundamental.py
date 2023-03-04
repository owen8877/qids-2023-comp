from unittest import TestCase

from xarray import Dataset
import xarray as xr


def calculate_fundamental_v0(ds: Dataset):
    earnings_ttm = ds['close_0'] / ds['pe_ttm']
    earnings = ds['close_0'] / ds['pe']
    book = ds['close_0'] / ds['pb']
    sales = ds['close_0'] / ds['ps']
    cashflow = ds['close_0'] / ds['pcf']

    market_share_history = ds['volume'].sum('timeslot') / ds['turnoverRatio']
    market_share = market_share_history.sel(day=slice(990, 1000)).median(dim='day') / 1e8
    market_cap = market_share * ds['close_0']

    return Dataset(data_vars={
        'earnings_ttm': earnings_ttm,
        'earnings': earnings,
        'book': book,
        'sales': sales,
        'cashflow': cashflow,
        'market_share': market_share,
        'market_cap': market_cap,
    })


class Test(TestCase):
    def test_export_fundamental_v0(self):
        path = '../data/nc'
        base_ds = xr.open_dataset(f'{path}/base.nc')
        market_brief_ds = xr.open_dataset(f'{path}/market_brief.nc')
        ds = base_ds.merge(market_brief_ds)

        fundamental_ds = calculate_fundamental_v0(ds)
        fundamental_ds.to_netcdf(f'{path}/fundamental_v0.nc')

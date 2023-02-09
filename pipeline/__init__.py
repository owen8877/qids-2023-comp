import pandas as pd
from pandas import DataFrame

N_asset = 54
N_timeslot = 50


class Dataset:
    def __init__(self, market: DataFrame, fundamental: DataFrame, ref_return: DataFrame):
        self.is_train = ref_return is not None

        self.market = market
        self.fundamental = fundamental
        self.ref_return = ref_return

    def dump(self, path: str):
        self.market.to_feather(f'{path}/market.feather')
        self.fundamental.to_feather(f'{path}/fundamental.feather')
        self.ref_return.to_feather(f'{path}/ref_return.feather')

    @staticmethod
    def load(path: str = 'data/parsed'):
        market = pd.read_feather(f'{path}/market.feather')
        fundamental = pd.read_feather(f'{path}/fundamental.feather')
        ref_return = pd.read_feather(f'{path}/ref_return.feather')

        return Dataset(market, fundamental, ref_return)

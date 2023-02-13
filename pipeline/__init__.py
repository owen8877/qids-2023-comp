from pathlib import Path
from unittest import TestCase

import pandas as pd
from pandas import DataFrame

from pipeline import parse_raw_df
from util import ensure_dir

N_asset = 54
N_timeslot = 50
N_train_days = 1000
N_test_days = 700


class Dataset:
    def __init__(self, market: DataFrame, fundamental: DataFrame, ref_return: DataFrame):
        self.is_train = ref_return is not None

        self.market = market
        self.fundamental = fundamental
        self.ref_return = ref_return

    def dump(self, path: str):
        ensure_dir(path)
        # Notice that the convention is to make multiindex ['day', 'asset', ?'timeslot'], but feather does not support
        # multiindex, thus we need to reset index.
        self.market.reset_index().to_feather(f'{path}/market.feather')
        self.fundamental.reset_index().to_feather(f'{path}/fundamental.feather')
        self.ref_return.reset_index().to_feather(f'{path}/ref_return.feather')

    @staticmethod
    def load(path: str = 'data/parsed'):
        try:
            market = pd.read_feather(f'{path}/market.feather').set_index(['day', 'asset', 'timeslot']).sort_index()
            fundamental = pd.read_feather(f'{path}/fundamental.feather').set_index(['day', 'asset']).sort_index()
            ref_return = pd.read_feather(f'{path}/ref_return.feather').set_index(['day', 'asset']).sort_index()
            return Dataset(market, fundamental, ref_return)
        except FileNotFoundError as e:
            print(f'Data load failed! Check if the pre-processed data is at {path}.')
            print('Hint: run the `Test.test_parse_raw_df_and_dump_full` method in `@/pipeline/parse_raw_df.py` if this'
                  'is first time to you.')
            raise e


def generate_mini_csv(n_days: int = 10):
    """
    Generate a mini dataset (usually consisting 10 days) that is useful for unittesting.

    :param n_days:
    :return:
    """
    mini_path = f'../data/raw_mini/{n_days}'
    full_path = f'../data/raw'
    ensure_dir(mini_path)
    template = '{}/first_round_train_{}_data.csv'
    try:
        pd.read_csv(template.format(full_path, 'fundamental')).iloc[:N_asset * n_days, :].to_csv(
            template.format(mini_path, 'fundamental'))
        pd.read_csv(template.format(full_path, 'market')).iloc[:N_asset * N_timeslot * n_days, :].to_csv(
            template.format(mini_path, 'market'))
        pd.read_csv(template.format(full_path, 'return')).iloc[:N_asset * n_days, :].to_csv(
            template.format(mini_path, 'return'))
    except FileNotFoundError as e:
        print('csv raw data not found! make sure that the training fundamental/market/return data '
              'is under the @/data/raw folder')
        raise e


def load_mini_dataset(parent_path: str = 'data/parsed_mini', n_days: int = 10):
    market_path = f'{parent_path}/{n_days}/market.feather'
    fundamental_path = f'{parent_path}/{n_days}/fundamental.feather'
    ref_return_path = f'{parent_path}/{n_days}/ref_return.feather'
    if not Path(market_path).exists():
        print(f'Warning: could not find mini market data at `{market_path}`, trying to run the pipleline.')
        generate_mini_csv(n_days)
        parse_raw_df._dump(is_mini=True, n_days=n_days)

    market = pd.read_feather(market_path).set_index(['day', 'asset', 'timeslot']).sort_index()
    fundamental = pd.read_feather(fundamental_path).set_index(['day', 'asset']).sort_index()
    ref_return = pd.read_feather(ref_return_path).set_index(['day', 'asset']).sort_index()
    return Dataset(market, fundamental, ref_return)


class Test(TestCase):
    def test_generate_mini_dataset(self):
        generate_mini_csv(n_days=10)

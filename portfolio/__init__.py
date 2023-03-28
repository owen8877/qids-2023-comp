from abc import abstractmethod
from datetime import datetime

from xarray import Dataset, DataArray

from util import ensure_dir


class Portfolio:
    def __init__(self, lookback_window: int, name: str):
        self.lookback_window = lookback_window
        self.name = name

    @abstractmethod
    def initialize(self, X: Dataset, y: DataArray):
        pass

    @abstractmethod
    def train(self, X: Dataset, y: DataArray):
        pass

    @abstractmethod
    def construct(self, X: Dataset) -> DataArray:
        pass

    def dump_stat(self, stat: Dataset, path_prefix='.', need_timestamp: bool = True):
        ensure_dir(f'{path_prefix}/stat')
        now = datetime.now()
        if need_timestamp:
            stat.to_netcdf(f'{path_prefix}/stat/{self.name}-{now.strftime("%Y%m%d-%H%M%S")}.nc')
        else:
            stat.to_netcdf(f'{path_prefix}/stat/{self.name}.nc')

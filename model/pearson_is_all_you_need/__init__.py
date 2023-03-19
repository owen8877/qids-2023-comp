from pathlib import Path

import numpy as np
import torch
import xarray as xr
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import RobustScaler
from torch import optim

from datatools import extract_market_data
from model.neural_network import NN_wrapper
from pipeline.backtest import evaluation_for_submission, cross_validation
from pipeline.fundamental import calculate_fundamental_v0
from pipeline.parse_raw_df import _dump
from qids_code_example.qids_lib import QIDS
from visualization.metric import plot_performance

##### Load necessary dataset and create them if not present
path = '../../data/nc'
base_nc_path = f'{path}/base.nc'
if not Path(base_nc_path).exists():
    _dump(is_mini=False, n_days=-1)
base_ds = xr.open_dataset(base_nc_path)

market_brief_path = f'{path}/market_brief.nc'
if not Path(market_brief_path).exists():
    market_brief = extract_market_data(base_ds[['money', 'volume', 'close', 'open', 'high', 'low']])
    market_brief.to_netcdf(market_brief_path)
market_brief_ds = xr.open_dataset(market_brief_path)
ds = base_ds.merge(market_brief_ds)

fundamental_path = f'{path}/fundamental_v0.nc'
if not Path(fundamental_path).exists():
    fundamental_ds = calculate_fundamental_v0(ds)
    fundamental_ds.to_netcdf(fundamental_path)
else:
    fundamental_ds = xr.open_dataset(fundamental_path)
ds = ds.merge(fundamental_ds)


##### Datatools extension
def winsorize(X, q=0.1):
    lower = X.quantile(q)
    upper = X.quantile(1 - q)
    return X.clip(lower, upper, axis=1)


class CombinedScaler:
    def __init__(self):
        self.scaler = RobustScaler()

    def fit_transform(self, X):
        X = winsorize(DataFrame(X))
        return self.scaler.fit_transform(X)

    def transform(self, X):
        X = winsorize(DataFrame(X))
        return self.scaler.transform(X)


##### CUDA
IS_CUDA = torch.cuda.is_available()
print(f'>>>> IS_CUDA={IS_CUDA} <<<<<')

##### Actually learning (?)
HIDDEN_SIZE = 8
train_lookback = 2
eval_lookback = 2
feature = ['book', 'cashflow', 'sales', 'earnings_ttm', 'earnings', 'market_cap']

preprocess = CombinedScaler()
lr = 5e-4
n_epoch = 2
constructor = lambda o: optim.lr_scheduler.CyclicLR(o, base_lr=lr / 10, max_lr=lr, step_size_up=n_epoch // 2,
                                                    cycle_momentum=False)

torch.manual_seed(2023)
np.random.seed(2023)

model = NN_wrapper(preprocess=preprocess, lr=lr, lr_scheduler_constructor=constructor, n_epoch=n_epoch,
                   train_lookback=train_lookback, per_eval_lookback=eval_lookback, hidden_size=HIDDEN_SIZE, n_asset=54,
                   network='Transformer', feature_name=feature, load_model_path=None, criterion='pearson',
                   embed_asset=True, embed_offset=True, is_eval=False, var_weight=0,
                   is_cuda=IS_CUDA)
ds_cv = ds.sel(day=slice(200, ds.dims['day'] - 2))[feature + ['return']]
performance_cv, cum_y_df = cross_validation(model, feature, ds=ds_cv, train_lookback=train_lookback,
                                            per_eval_lookback=eval_lookback)

plt.figure(0)
plot_performance(performance_cv, metrics_selected=['val_cum_pearson', 'val_cum_r2'])
plt.show()

##### Evaluation
qids = QIDS(path_prefix='../../')

load_model_path = model.model_path
preprocess = CombinedScaler()
lr = 5e-4
n_epoch = 1
# constructor = lambda o: optim.lr_scheduler.CyclicLR(o, base_lr=lr/100, max_lr=lr, step_size_up=n_epoch // 2,
#                                                     cycle_momentum=False)
torch.manual_seed(2023)
np.random.seed(2023)

model = NN_wrapper(preprocess=preprocess, lr=lr, lr_scheduler_constructor=None, n_epoch=n_epoch,
                   train_lookback=train_lookback, per_eval_lookback=eval_lookback, hidden_size=HIDDEN_SIZE, n_asset=54,
                   network='Transformer', feature_name=feature, load_model_path=load_model_path,
                   embed_asset=True, embed_offset=True, criterion='pearson',
                   is_eval=True,
                   is_cuda=IS_CUDA)
performance_eval, y_pred = evaluation_for_submission(model, given_ds=ds, lookback_window=train_lookback,
                                                     per_eval_lookback=eval_lookback, qids=qids)
plt.figure(0)
plot_performance(performance_eval, metrics_selected=['test_cum_pearson', 'test_cum_r2'])
plt.show()

from datetime import date
from typing import Optional, Callable, Tuple
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from pandas import Series, DataFrame
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR

import torch.nn.functional as F
import xarray as xr
from xarray import Dataset, DataArray

from util import ensure_dir
from torchmetrics import PearsonCorrCoef

idx = pd.IndexSlice


def _num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def _device_helper(is_cuda):
    return 'cuda' if is_cuda else 'cpu'


class CUDAModule(nn.Module):
    def __init__(self, is_cuda: bool):
        super().__init__()
        self.is_cuda = is_cuda
        self.device = _device_helper(self.is_cuda)
        self._has_moved = False

    def move(self):
        self._has_moved = True
        self.to(self.device)

    def forward_device_independent(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        if not self._has_moved:
            print('This module might have not been moved to the correct device!')

        return self.forward_device_independent(x.to(self.device)).to('cpu')


class MLP(CUDAModule):
    """
    Fully connected NN
    """

    def __init__(self, D_out, input_shape, is_cuda: bool = True):
        """

        :param D_out: number of classes label
        :param input_shape:
        :param is_cuda:
        """
        super(MLP, self).__init__(is_cuda)

        self.fc1 = nn.Linear(input_shape[1] * input_shape[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, D_out)
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        # self.final_layer = nn.Softmax()

        self.move()

    def forward_device_independent(self, x):
        x = x.contiguous().view(-1, _num_flat_features(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.dropout(self.fc3(x))))
        x = self.fc4(x)
        # x = self.final_layer(x)
        return x


class LSTM(CUDAModule):
    def __init__(self, num_output, num_features, hidden_size, num_layers, is_cuda: bool = False):
        """

        Remark:
            1. hidden_size is like embedding feature space dimension
            2. better > num_assets?
                # if num_layers = 2: stack 2 LSTM of lyaer 1:
                # nn.Sequential(OrderedDict([
                #     ('LSTM1', nn.LSTM(input_size, hidden_size, 1),
                #     ('LSTM2', nn.LSTM(hidden_size, hidden_size, 1)
                #     ]))
        :param num_output:
        :param num_features:
        :param hidden_size:
        :param num_layers:
        :param is_cuda:
        """
        super(LSTM, self).__init__(is_cuda)
        self.num_output = num_output
        self.num_layers = num_layers
        self.num_features = num_features
        self.hidden_size = hidden_size
        # LSTM input size (batch_first) (batch, seq_length, feature)
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, num_output)

        self.move()

    def forward_device_independent(self, x: torch.Tensor):
        """

        Remark: the second dim for h_0 and c_0 is the batch dim
        :param x:
        :return:
        """
        batch_size = x.shape[0]
        # TODO: review: Variable if deprecated
        # https://pytorch.org/docs/stable/autograd.html?highlight=variable#variable-deprecated
        h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device)  # hidden state
        c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, device=self.device)  # cell state

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        # get only the last hidden layer, needed for multiple layer
        h_out = h_out[-1, :, :].view(-1, self.hidden_size)

        out = self.fc(self.dropout(h_out))

        return out


class oneDVerConvNet(CUDAModule):
    def __init__(self, D_in, D_out, input_shape, b_size, is_cuda: bool = True, activation: str = 'relu'):
        # input_shape: without batch dimension
        # b_size: batch size
        super(oneDVerConvNet, self).__init__(is_cuda)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.layer1 = nn.Sequential(
            nn.Conv2d(D_in, 16, (3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(16),  # no batch
            self.activation,
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
            self.activation,
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 1024, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.BatchNorm2d(1024),
            self.activation,
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.drop_out = nn.Dropout()
        n_size = self._get_conv_output(input_shape, b_size)
        self.fc1 = nn.Linear(n_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, D_out)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)

        self.move()

    def _get_conv_output(self, shape, b_size):
        input = Variable(torch.rand(b_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(b_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape) # shape = (b_size, feature=1024, H=1, W = 54)
        x = torch.max(x, 3)[0]
        return x

    def forward_device_independent(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.max(x, 3)[0]  # questionable maybe do along [1]

        x = x.view(-1, _num_flat_features(x))
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.drop_out(self.fc2(x))))
        x = self.fc3(x)
        # x = F.relu(self.transform1(x))
        # x = F.relu(self.drop_out(self.transform2(x)))
        # x = self.transform3(x)
        return x.squeeze()


class Transformer(CUDAModule):
    def __init__(self, n_class: int, seq_length: int, feature: list, hidden_size: int, output_size: int,
                 num_layer: int = 1, is_cuda: bool = False, offset_feature_size: Optional[int] = None):
        super(Transformer, self).__init__(is_cuda)
        self.n_feature = len(feature)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer

        self.class_dim = n_class
        self.time_dim = seq_length
        # self.class_embed = nn.Embedding(self.class_dim, self.) # try without embedding first

        self.embed_feature_dim = self.n_feature  # no embedding
        ## one-hot encoding and positional encoding:
        if 'offset_feature' in feature:
            self.is_time_embed = True
            self.time_embed = nn.Embedding(offset_feature_size, 8)
            self.embed_feature_dim += 7
            self.day_feature_indx = feature.index('offset_feature')
        else:
            self.is_time_embed = False

        if 'asset_name' in feature:
            self.embed_feature_dim += 53
            self.is_asset_embed = True
            self.asset_name_indx = feature.index('asset_name')
        else:
            self.is_asset_embed = False

        if 'season_day' in feature:
            self.is_season_embed = True
            self.season_embed = nn.Embedding(65, 8)
            self.embed_feature_dim += 7
            self.season_day_indx = feature.index('season_day')

        print('embedding feature dimension is', self.embed_feature_dim)

        self.linear_1 = nn.Linear(self.embed_feature_dim, self.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                   nhead=8,
                                                   dropout=0.0,
                                                   batch_first=True,
                                                   )
        self.encoder = nn.TransformerEncoder(encoder_layer, self.num_layer)
        self.linear_2 = nn.Linear(self.hidden_size, self.output_size)

        self.move()

    def forward_device_independent(self, x: torch.Tensor):
        columns_indx = torch.arange(x.shape[-1]).long()
        # print('original indx', columns_indx)
        indx_remain_bool = torch.ones(columns_indx.shape).long()

        if self.is_time_embed:
            day_feature = x[:, :, self.day_feature_indx].long()
            day_feature = self.time_embed(day_feature)
            indx_remain_bool *= (columns_indx != self.day_feature_indx)

        if self.is_asset_embed:
            # print(self.asset_name_indx)
            asset_name = x[:, :, self.asset_name_indx]
            # print(x)
            # print(asset_name)
            # print(asset_name.max())
            assert asset_name.max().item() <= 54
            asset_name = F.one_hot(asset_name.long(), 54)
            indx_remain_bool *= (columns_indx != self.asset_name_indx)

        if self.is_season_embed:
            season_day = x[:, :, self.season_day_indx].long()
            season_day = self.season_embed(season_day)
            indx_remain_bool *= (columns_indx != self.season_day_indx)

        # print('remain indices:', columns_indx[indx_remain_bool.long()==1])

        # get data out except for embedded ones
        x = x[:, :, indx_remain_bool.long() == 1]
        # print(x.shape)
        if self.is_time_embed:
            x = torch.cat((x, day_feature), dim=-1)
        if self.is_asset_embed:
            x = torch.cat((x, asset_name), dim=-1)
        if self.is_season_embed:
            x = torch.cat((x, season_day), dim=-1)

        # print(x.shape) # (432, 4, 71)

        x = self.linear_1(x)
        h = self.encoder(x)
        h = h.mean(dim=1)
        outputs = self.linear_2(h)
        return outputs.squeeze()


class SimpleLinearNetwork(nn.Module):
    def __init__(self, D_in):
        super(SimpleLinearNetwork, self).__init__()
        self.fc = nn.Linear(D_in, 1)

    def forward(self, x):
        x = torch.transpose(x[:, :, 0, :], 1, 2)
        return self.fc(x).squeeze()


def easy_winsorize(s: Series, trim_ratio: float = 0.05):
    return np.clip(s, s.quantile(trim_ratio), s.quantile(1 - trim_ratio))


class DataPreprocessing:
    def __init__(self):
        self.scaler = RobustScaler()

    def unskew(self, X: DataFrame):
        return DataFrame({
            'log_turnoverRatio': np.log(X['turnoverRatio']),
            'log_transactionAmount_pos': np.log(X['transactionAmount'] + 1),
            'log_pb': np.log(X['pb']),
            'log_ps': np.log(X['ps']),
            # 'winsor_pe_ttm': easy_winsorize(X['pe_ttm']),
            'winsor_pe': easy_winsorize(X['pe']),
            'winsor_pcf': easy_winsorize(X['pcf']),
        })

    def working(self, X: DataFrame, t):
        # return DataFrame(t(self.unskew(X)), index=X.index, columns=list(set(X.columns) - {'return_0+1'})
        #                  ).merge(X['return_0+1'], on=['day', 'asset'])
        tX = self.unskew(X)
        aa = (X['return_0'] + 1) * (X['return_1'] + 1) - 1
        df = DataFrame(t(tX), index=tX.index, columns=tX.columns)
        df['return_0+1'] = aa
        return df

    def fit_transform(self, X: DataFrame):
        return self.working(X, lambda Y: self.scaler.fit_transform(Y))

    def transform(self, X: DataFrame):
        return self.working(X, lambda Y: self.scaler.transform(Y))


def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Neg_Pearson_Loss(nn.Module):
    def __init__(self) -> None:
        super(Neg_Pearson_Loss, self).__init__()

    def forward(self, pred, target):
        # loss_without_reduction = max(0, −target * (input1 − input2) + margin)
        pearson = PearsonCorrCoef()
        return -pearson(pred.squeeze(), target.squeeze())


class MonStEr_Loss(nn.Module):
    def __int__(self) -> None:
        super(MonStEr_Loss, self).__int__()

    def forward(self, pred_, target_):
        pred = pred_/0.2
        target = target_/0.2
        loss = 0
        # is_tar_pos = target >= 0
        # is_pred_ge_tar = pred >= target

        is_dir_correct= torch.sign(pred-target) == torch.sign(target)
        # print(target.max())
        loss += torch.sum(is_dir_correct*(1/(1+0.1*torch.abs(target)))*torch.abs(pred-target)\
             + (~is_dir_correct)*torch.pow(torch.abs(target - pred), 1+4*torch.abs(target)))


        # loss += torch.sum(is_tar_pos * is_pred_ge_tar * (1 / (1 + 4 * target) * (pred - target)) \
        #         + is_tar_pos * (~is_pred_ge_tar) * torch.pow(target - pred, 1 + target) \
        #         + (~is_tar_pos) * is_pred_ge_tar * torch.pow(target - pred, 1 - target) \
        #         + (~is_tar_pos) * (~is_pred_ge_tar) * (1 / (1 + 4 * target) * (target - pred)))

        return loss


class NN_wrapper:
    def __init__(
            self, preprocess, lr=0.001, criterion=nn.MSELoss(), n_epoch=5, train_lookback=32, per_eval_lookback=16,
            n_asset=54, hidden_size=64, lr_scheduler_constructor: Optional[Callable] = None, network='LSTM',
            is_eval: bool = False, feature_name=None, load_model_path=None, is_cuda=True, embed_season: bool = False,
            embed_offset: bool = False, embed_asset: bool = False, l2_weight=1e-5, l1_weight=0, var_weight=0
    ):
        if feature_name is None:
            raise ValueError('Please provide a list of feature')
        feature_to_nn = feature_name.copy()
        if embed_offset:
            feature_to_nn += ['offset_feature']
        if embed_asset:
            feature_to_nn += ['asset_name']
        n_feature = len(feature_name)

        if network == 'LSTM':
            self.net = LSTM(num_output=1, num_features=n_feature, hidden_size=hidden_size, num_layers=1,
                            is_cuda=is_cuda)
        elif network == 'MLP':
            self.net = MLP(D_out=1, input_shape=[n_asset * train_lookback, per_eval_lookback, n_feature],
                           is_cuda=is_cuda)
        elif network == 'CONV':
            self.net = oneDVerConvNet(D_in=1, D_out=1, input_shape=[1, per_eval_lookback, n_feature],
                                      b_size=train_lookback * n_asset, is_cuda=is_cuda)
        elif network == 'Transformer':
            self.net = Transformer(n_class=n_asset, seq_length=per_eval_lookback, feature=feature_to_nn,
                                   hidden_size=hidden_size, output_size=1, is_cuda=is_cuda,
                                   offset_feature_size=per_eval_lookback)
        else:
            raise ValueError('Network architecture not supported')

        if is_eval:
            if load_model_path is None:
                raise ValueError('Please provide model path to load')
            else:
                self.net.load_state_dict(torch.load(load_model_path))

        if load_model_path is not None:
            self.net.load_state_dict(torch.load(load_model_path))

        self.is_eval = is_eval
        self.feature_name = feature_name
        self.feature_to_nn = feature_to_nn
        self.n_epoch = n_epoch

        if criterion == 'pearson':
            self.criterion = Neg_Pearson_Loss()
        elif criterion == 'monster':
            self.criterion = MonStEr_Loss()
        elif criterion == 'combined':
            self.criterion = 'combined'
        else:
            self.criterion = criterion

        self.train_lookback = train_lookback
        self.per_eval_lookback = per_eval_lookback
        self.n_asset = n_asset
        self.net_name = network
        self.embed_asset = embed_asset
        self.embed_offset = embed_offset
        self.embed_season = embed_season

        # regularization parameter
        self.l1_weight = l1_weight
        self.var_weight = var_weight

        # Define the optimizier
        ## optimizer with L2-regularization
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=l2_weight)

        ## original optimizer
        # self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999))

        # self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr / 10, max_lr=lr,
        #                                              step_size_up=n_epoch // 2, cycle_momentum=False)
        if lr_scheduler_constructor is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.5)
        else:
            self.scheduler = lr_scheduler_constructor(self.optimizer)

        self.preprocess = preprocess
        self.is_learning = True
        # self.early_stopper = EarlyStopper(patience=3, min_delta=10)

    def prepare_X(self, X: Dataset, fit: bool, lookback: int) -> Tuple[torch.Tensor, DataArray]:
        start_day = X.day.min().to_numpy().item()

        X_pd = X[self.feature_name].to_dataframe(dim_order=['day', 'asset'])

        ## Minmax scale except for the asset name (category)
        X_pd[X_pd.columns] = self.preprocess.fit_transform(X_pd.values) if fit else self.preprocess.transform(
            X_pd.values)
        X_transformed = xr.Dataset.from_dataframe(X_pd)

        # New data preprocessing with Xarray - Hooray!
        X_list = []
        for i in range(lookback):
            X_slice = X_transformed.sel(
                day=slice(start_day + i, start_day + i + self.per_eval_lookback - 1)).expand_dims(
                batch=[start_day + i + self.per_eval_lookback - 1])
            X_slice.coords['offset'] = X_slice.day - start_day - i  # calculate offset coordinate/index for each slice
            X_slice_o = X_slice.swap_dims(
                {'day': 'offset'})  # swap to make offset the dimension instead of the previous 'day'
            X_list.append(X_slice_o.reset_coords(drop=True))  # (reset and) drop all non-index coordinates

        X_concat = xr.concat(X_list, dim='batch')  # concat along the batch dimension
        if self.embed_offset:
            X_concat = X_concat.assign(offset_feature=X_concat.offset)
        if self.embed_asset:
            X_concat = X_concat.assign(asset_name=X_concat.asset)
        if self.embed_season:
            X_concat = X_concat.assign(season_day=X_concat.day)
            X_concat['season_day'] = X_concat['season_day'] % 65

        # create a new batch dimension cartesian product all batch and asset
        X_arr = (X_concat.stack({'batch_asset': ['batch', 'asset']}).to_array('feature')
                 .transpose('batch_asset', 'offset', 'feature'))
        X_arr_np = X_arr.to_numpy()

        if self.net_name == 'CONV':
            X_arr_np = X_arr_np[:, np.newaxis, :, :]

        # then reshape into the desire form for NN (batch, seq_length, feature)
        # batch_asset "is" a multi-index array, with the ordering of train_lookback, asset
        X_tensor = torch.from_numpy(X_arr_np).to(torch.float)
        return X_tensor, X_arr.batch_asset

    def fit_predict(self, X: Dataset, y: DataArray):
        self.net.train()
        X_tensor, batch_asset = self.prepare_X(X, fit=True, lookback=self.train_lookback)
        y_tensor = torch.from_numpy(y.stack({'batch_asset': ['day', 'asset']}).values).to(torch.float)

        # LSTM or NN shape (batch, seq_length, feature)
        for epoch in range(self.n_epoch):
            self.optimizer.zero_grad()
            outputs = self.net(X_tensor).squeeze()

            if self.criterion == 'combined':
                mse_loss = nn.MSELoss()
                pearson_loss = Neg_Pearson_Loss()
                loss = 0.1*mse_loss(outputs, y_tensor) + pearson_loss(outputs, y_tensor)
            else:
                loss = self.criterion(outputs, y_tensor)

            if epoch == 1:
                print('training loss:', loss.item())

            ## add L1 regularization
            if self.l1_weight > 0:
                l1_norm = sum(torch.linalg.norm(p, 1) for p in self.net.parameters())
                loss += self.l1_weight * l1_norm

            if self.var_weight > 0 and isinstance(self.criterion, nn.MSELoss):
                # print('variance is computed')
                loss += self.var_weight * torch.abs(torch.var(outputs) - torch.var(y_tensor))

            ## check parameter values
            # print(list(self.net.parameters()))

            ## check parameter name and require_grad
            # for name, param in self.net.named_parameters():
            #     print(name, param.requires_grad)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        if X.day.max().to_numpy().item() >= 994 and not self.is_eval:  # save final model
            dump_folder = 'model/dump/' + str(date.today())
            model_path = dump_folder + '/' + str(date.today()) + '_' + self.net_name
            ensure_dir(dump_folder)
            torch.save(self.net.state_dict(), model_path)
            print('Final learning rate:', _get_lr(self.optimizer))

        return xr.DataArray(data=outputs.detach().numpy(), coords=dict(batch_asset=batch_asset)).unstack(
            'batch_asset').rename({'batch': 'day'})

    def predict(self, X):
        self.net.eval()
        X_tensor, _ = self.prepare_X(X, fit=False, lookback=1)
        y = self.net(X_tensor).squeeze()
        # return np.clip(y.detach().numpy(), -0.2, 0.2)[np.newaxis, :]  # return a numpy array
        return y.detach().numpy()[np.newaxis, :]  # for predicting correlation


class Test(TestCase):
    def test_nn(self):
        """
        Adapted from `playground/disaspeare/Feb_17_NN_MLP.ipynb`

        :return:
        """
        from pipeline import Dataset
        import pandas as pd
        from torch import nn
        from datatools import check_dataframe
        from pipeline.backtest import cross_validation
        from matplotlib import pyplot as plt
        from visualization.metric import plot_performance

        dataset = Dataset.load('../data/parsed')
        df = pd.concat([dataset.fundamental, dataset.ref_return], axis=1).dropna()

        check_dataframe(df, expect_index=['day', 'asset'])
        df['return_0+1'] = df['return'].shift(2 * 54).fillna(0)

        train_lookback = 64
        eval_lookback = 16
        n_epoch = 100
        lr = 0.001
        criterion = nn.MSELoss()
        # optimizer = 'LBFGS'
        optimizer = 'ADAM'
        original_feature = ['turnoverRatio', 'transactionAmount', 'pb', 'ps', 'pe', 'pcf', 'return_0+1']
        n_asset = 54

        pp = DataPreprocessing()
        my_nn = NN_wrapper(pp, None, eval_lookback, train_lookback, optimizer=optimizer, learning_rate=lr,
                           criterion=criterion, n_epoch=n_epoch, n_asset=n_asset, n_feature=len(original_feature))
        performance, pred_df = cross_validation(my_nn, feature_columns=original_feature,
                                                df=df.query(f'asset < {n_asset}'),
                                                per_eval_lookback=eval_lookback, train_lookback=train_lookback)

        plt.figure()
        plot_performance(performance, metrics_selected=['train_r2', 'val_cum_r2', 'val_cum_pearson'])
        plt.ylim([-0.15, 0.1])
        plt.show()

    def test_monster(self):
        import torch

        a=torch.rand(10)
        b=torch.rand(10)
        criterion = MonStEr_Loss()
        loss = criterion(a,b)
        print(a,'\n')
        print(b, '\n')
        print(loss)
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

import xarray as xr

idx = pd.IndexSlice

FEATURE_1 = 12
FEATURE_2 = 12
FEATURE_3 = 12
FEATURE_4 = 12

FC_3 = 8
FC_2 = 5


class oneDVerConvNet(nn.Module):  # type3 - 1D Vertical Convolution
    def __init__(self, D_in, D_out, input_shape, b_size):
        # input_shape: without batch dimension
        # b_size: batch size
        super(oneDVerConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(D_in, FEATURE_1, (3, 1), stride=(1, 1), padding=(1, 0)),
            # nn.BatchNorm2d(16), # no batch
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(FEATURE_1, FEATURE_2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            # nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(FEATURE_2, FEATURE_3, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            # nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.layer4 = nn.Sequential(
            nn.Conv2d(FEATURE_3, FEATURE_4, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            # nn.BatchNorm2d(1024),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        self.drop_out = nn.Dropout()
        n_size = self._get_conv_output(input_shape, b_size)
        # print(n_size)
        #       self.fc1 = nn.Linear(n_size, 256)
        self.fc1 = nn.Linear(n_size, FC_3)
        self.fc2 = nn.Linear(FC_3, FC_2)
        self.fc3 = nn.Linear(FC_2, D_out)
        # self.transform1 = nn.Conv2d(n_size, FEATURE_3, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.transform2 = nn.Conv2d(FEATURE_3, FEATURE_2, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.transform3 = nn.Conv2d(FEATURE_2, FEATURE_1, kernel_size=(1, 1), stride=(1, 1), padding=0)
        # self.bn1 = nn.BatchNorm1d(256)
        # self.bn2 = nn.BatchNorm1d(64)

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

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = torch.max(x,3)[0] # questionable maybe do along [1]
        # print(x.shape) # shape = (b_size, feature=1024, H=1) after max
        # print(x.data.size())
        # x = x.view(-1,self.num_flat_features(x))
        #        print(x.data.size())
        #         x = F.relu(self.bn1(self.fc1(x)))
        #         x = F.relu(self.bn2(self.drop_out(self.fc2(x))))
        x = torch.transpose(x[:, :, 0, :], 1, 2)
        x = torch.tanh(self.drop_out(self.fc1(x)))
        x = torch.tanh(self.drop_out(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(dim=-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class oneD16_4_1VerConvNet(nn.Module):
    def __init__(self, D_in, D_out, input_shape, b_size):
        # input_shape: without batch dimension
        # b_size: batch size
        super(oneD16_4_1VerConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(D_in, FEATURE_1, (3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(FEATURE_1, FEATURE_2, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)))
        self.drop_out = nn.Dropout()
        n_size = self._get_conv_output(input_shape, b_size)
        self.fc1 = nn.Linear(n_size, FC_3)
        self.fc2 = nn.Linear(FC_3, FC_2)
        self.fc3 = nn.Linear(FC_2, D_out)

    def _get_conv_output(self, shape, b_size):
        input = Variable(torch.rand(b_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(b_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.max(x, 3)[0]
        return x

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.transpose(x[:, :, 0, :], 1, 2)
        x = torch.tanh(self.drop_out(self.fc1(x)))
        x = torch.tanh(self.drop_out(self.fc2(x)))
        x = self.fc3(x)
        x = x.squeeze(dim=-1)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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


class NN_wrapper():
    def __init__(self, preprocess, net_path, per_eval_lookback, train_lookback, optimizer,
                 learning_rate=0.003, criterion=nn.MSELoss(), n_epoch=3, n_asset=54, n_feature=7):

        self.preprocess = preprocess
        self.n_asset = n_asset
        self.net = oneD16_4_1VerConvNet(D_in=n_feature, D_out=1,
                                        input_shape=(n_feature, per_eval_lookback, self.n_asset), b_size=1)
        if net_path:
            self.net.load_state_dict(torch.load(net_path))
        self.per_eval_lookback = per_eval_lookback
        self.train_lookback = train_lookback
        self.learning_rate = learning_rate
        self.criterion = criterion
        if optimizer == 'LBFGS':
            self.optimizer = optim.LBFGS(self.net.parameters(), lr=learning_rate)
        elif optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        else:
            raise ValueError('Optimizer not supported')
        # self.lbfgs = optim.LBFGS(self.net.parameters(), lr=learning_rate, max_iter=10, max_eval=100)
        # self.adam = optim.Adam(self.net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        self.lscheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.ascheduler = ExponentialLR(self.optimizer, gamma=0.999)
        self.optim_name = optimizer
        self.n_epoch = n_epoch
        self.is_learning = True

    def fit_predict(self, X_, y_):
        self.net.train()
        X = xr.Dataset.from_dataframe(X_)
        y = xr.DataArray.from_series(y_)
        X_pd = X.to_dataframe(dim_order=['day', 'asset'])
        X_transformed_pd = self.preprocess.fit_transform(X_pd)
        X_transformed = xr.Dataset.from_dataframe(X_transformed_pd)

        start_day = X_transformed.day.min().to_numpy().item()

        X_ult = torch.cat([torch.from_numpy(
            X_transformed.sel(day=slice(start_day + i, start_day + i + self.per_eval_lookback - 1))
            .to_array(dim='feature').transpose('feature', 'day', 'asset')
            .to_numpy()[np.newaxis, :]) for i in range(self.train_lookback)
        ], dim=-1).to(torch.float)
        y_ult = torch.cat([torch.from_numpy(
            y.sel(day=start_day + i + self.per_eval_lookback - 1)
            .to_numpy()[np.newaxis, :]) for i in range(self.train_lookback)
        ], dim=-1).to(torch.float)

        for epoch in range(self.n_epoch):
            # print('its actually training')
            def closure():
                self.optimizer.zero_grad()
                outputs = self.net(X_ult)
                loss = self.criterion(outputs, y_ult)
                loss.backward()
                return loss

            self.optimizer.zero_grad()
            outputs = self.net(X_ult)
            loss = self.criterion(outputs, y_ult)
            # print(loss)
            loss.backward()
            if self.optim_name == 'LBFGS':
                # if use_lbfgs:
                self.optimizer.step(closure)  # need closure for LBFGS
            else:
                self.optimizer.step()

        if self.optim_name == 'ADAM':
            self.ascheduler.step()
        else:
            self.lscheduler.step()
            # print(f'Epoch={epoch}, loss={cum_loss}')
        return outputs.detach().numpy().reshape(-1, 1)[:, 0]

    def predict(self, X):
        self.net.eval()
        X_transformed = self.preprocess.transform(X)
        X_np = X_transformed.swaplevel(1, 0).sort_index(ascending=True).to_numpy().astype(np.float32)
        # shape (asset, days, feature) -> (ft, days, asset)
        X_np_tensor = X_np.reshape(self.n_asset, self.per_eval_lookback, -1).transpose([2, 1, 0])
        X_np_tensor = X_np_tensor[np.newaxis, :]  # add batch dimension
        X_torch = torch.from_numpy(X_np_tensor)
        y = self.net(X_torch)
        # return np.clip(y.detach().numpy(), -0.2, 0.2)
        return y.detach().numpy().reshape(-1, 1)[:, 0]


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

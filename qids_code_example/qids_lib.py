import pandas as pd

TEST_MARKET_PATH = '../data/raw/first_round_test_market_data.csv'
TEST_FUNADMENTAL_PATH = '../data/raw/first_round_test_fundamental_data.csv'


class DataFeeder:
    def __init__(self, f_df, m_df):
        self.__current_idx = 0
        self.__predict_idx = 0
        self.__num_of_stocks = 54
        self.__point_per_day = 50
        self.__end = False
        self.__current_fundamental_df = None

        self.f_df = f_df
        self.m_df = m_df
        self.__length = len(self.f_df) / self.__num_of_stocks

    def is_end(self):
        return self.__end

    def get_current_market(self):
        if self.__end:
            raise ValueError('The environment has ended.')

        # check if the current index is equal to the predict index
        if self.__current_idx != self.__predict_idx:
            raise ValueError('The current index is not equal to the predict index.')

        # load data of the current day
        fundamental_df = self.f_df.iloc[
                         self.__current_idx * self.__num_of_stocks: (self.__current_idx + 1) * self.__num_of_stocks]
        market_df = self.m_df.iloc[self.__current_idx * self.__num_of_stocks * self.__point_per_day
                                   :(self.__current_idx + 1) * self.__num_of_stocks * self.__point_per_day]

        # update the current index
        self.__current_idx += 1
        self.__current_fundamental_df = fundamental_df.reset_index()

        return fundamental_df, market_df

    def input_prediction(self, predict_ds: pd.Series):
        if self.__end:
            raise ValueError('The environment has ended.')

        # check if the current index is equal to the predict index plus 1
        if self.__current_idx != self.__predict_idx + 1:
            raise ValueError('The current index is not equal to the predict index plus 1.')

        # check the length of the predict_ds
        if len(predict_ds) != self.__num_of_stocks:
            raise ValueError('The length of input decisions is wrong.')

        # check the type of the predict_ds
        if type(predict_ds) != pd.Series:
            raise TypeError('The type of input decisions is wrong.')

        self.__predict_idx += 1
        if self.__predict_idx == self.__length:
            self.__end = True
            print('Data Feeding is finished.')


class QIDS:
    def __init__(self, path_prefix: str = './'):
        self.__num_of_stocks = 54
        self.__point_per_day = 50

        self.__fundamental_df = pd.read_csv(path_prefix + TEST_FUNADMENTAL_PATH)
        self.__market_df = pd.read_csv(path_prefix + TEST_MARKET_PATH)

        if len(self.__fundamental_df) / self.__num_of_stocks != len(
                self.__market_df) / self.__num_of_stocks / self.__point_per_day:
            raise ValueError('The length of fundamental data and market data is not equal.')

    def make_env(self):
        return DataFeeder(self.__fundamental_df, self.__market_df)

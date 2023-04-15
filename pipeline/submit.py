from unittest import TestCase


class Test(TestCase):
    def test_submit(self):
        import requests

        # ===============================================================================================================================================================
        # Some inputs that you need to adjust
        STRATEGY_PATH = r"strategy.py"  # TODO: Change to your file address for strategy.py
        REQUIREMENTS_PATH = r"requirements.txt"  # TODO: Change to your file address for requirement.txt
        GROUP_ID = "G104"  # TODO: Change to "G" + group_id
        ACCESS_TOKEN = "OKZERvJdIGIbTtIi"  # TODO: Change to access_token
        # ===============================================================================================================================================================

        URL = "http://competition.hkuqids.com:8880/upload_submission"
        files = {
            "strategy": open(STRATEGY_PATH, "rb"),
            "requirements": open(REQUIREMENTS_PATH, "rb")
        }
        values = {
            "access_token": ACCESS_TOKEN,
            "group_id": GROUP_ID
        }
        input('Submit?')
        input('Really submit?')
        input('ARE YOU SURE?')
        r = requests.post(URL, files=files, data=values)
        print(r.text)

    def test_current_strategy(self):
        import numpy as np
        import pandas as pd
        from strategy import get_decisions
        from config import Path
        from tqdm.auto import trange
        import matplotlib.pyplot as plt
        days_for_train = Path.days_for_train

        market_df = pd.read_csv(f'{Path.historical_data_path}/second_round_all_market_data.csv')
        fundamental_df = pd.read_csv(f'{Path.historical_data_path}/second_round_all_fundamental_data.csv')

        days_end = len(fundamental_df) // 54
        # days_end = 200
        transactions = dict()
        holding_return = dict()
        open_fee = dict()
        close_fee = dict()

        pbar = trange(days_for_train + 1, days_end + 1)
        for day in pbar:
            # Clearing stage
            transaction_1 = transactions[day - 1] if day - 1 in transactions else np.zeros(54)
            transaction_2 = transactions[day - 2] if day - 2 in transactions else np.zeros(54)

            end_index = day * 50 * 54
            return_0 = market_df.iloc[end_index - 54:end_index]['close'].to_numpy() / \
                       market_df.iloc[end_index - 54 - 50 * 54:end_index - 50 * 54]['close'].to_numpy() - 1
            holding_return[day] = return_0 * (transaction_1 + transaction_2) / 2

            to_construct = day <= days_end - 2

            if to_construct:
                mm_df = market_df.iloc[end_index - 50 * 54:end_index].copy()
                mm_df['date'] = day
                ff_df = fundamental_df.iloc[day * 54 - 54:day * 54].copy()
                ff_df['date'] = day
                tr = get_decisions(mm_df, ff_df)
                # tr = np.random.rand(54)
                # tr /= tr.sum()
                assert len(tr) == 54
                assert isinstance(tr, list)
                tr = np.array(tr)
            else:
                tr = np.zeros(54)
            transactions[day] = tr
            transaction_0 = tr

            # Post-transaction fee calculation
            marginal_transaction = transaction_0 - transaction_2
            open_fee[day] = 4e-4 * np.clip(marginal_transaction, 0, None) / 2
            close_fee[day] = 2e-3 * np.clip(marginal_transaction, None, 0) / 2

        pnl = (np.array([holding_return[day] for day in range(days_for_train + 1, days_end + 1)])[1:, :]
               - np.array([open_fee[day] for day in range(days_for_train + 1, days_end + 1)])[:-1, :]
               - np.array([close_fee[day] for day in range(days_for_train + 1, days_end + 1)])[1:, :])

        plt.plot(np.cumsum(np.log(np.sum(pnl, axis=1) + 1)))
        plt.show()

    def test_generate_mock_dataset(self):
        from util import ensure_dir
        from config import Path
        import pandas as pd
        days_for_train = Path.days_for_train

        second_round_path = '../data/second_round_datasets'
        mock_data_path = f'../data/mock{days_for_train}'
        ensure_dir(mock_data_path)

        market_df = pd.read_csv(f'{second_round_path}/second_round_train_market_data.csv')
        market_df.iloc[:days_for_train * 50 * 54, :].set_index('date_time').to_csv(f'{mock_data_path}/second_round_train_market_data.csv')
        market_df.set_index('date_time').to_csv(f'{mock_data_path}/second_round_all_market_data.csv')

        f_df = pd.read_csv(f'{second_round_path}/second_round_train_fundamental_data.csv')
        f_df.iloc[:days_for_train * 54, :].set_index('date_time').to_csv(f'{mock_data_path}/second_round_train_fundamental_data.csv')
        f_df.set_index('date_time').to_csv(f'{mock_data_path}/second_round_all_fundamental_data.csv')

        r_df = pd.read_csv(f'{second_round_path}/second_round_train_return_data.csv')
        r_df.set_index('date_time').iloc[:(days_for_train - 2) * 54, :].to_csv(f'{mock_data_path}/second_round_train_return_data.csv')

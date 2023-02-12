from pandas import DataFrame
from datetime import datetime
import mplfinance as mpl
import matplotlib.pyplot as plt


def plot_price_volume(df: DataFrame, asset: int, day: int):
    new_df = df.loc[(df.asset == asset) & (df.day == day)]
    new_df = new_df[['open', 'high', 'low', 'close', 'volume']]
    new_df['datetime'] = new_df.index.map(lambda x: datetime.fromtimestamp(x))
    new_df = new_df.set_index('datetime')
    plt.figure(figsize=(20, 10))
    mpl.plot(new_df, type='candle', volume=True, mav=(5, 10, 20))
    plt.show()
    return None



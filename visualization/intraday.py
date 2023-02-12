from pandas import DataFrame
import mplfinance as mpl
import matplotlib.pyplot as plt


def plot_price_volume(df: DataFrame, asset: int, day: int):
    new_df = df.loc[(df.asset == asset) & (df.day == day)]
    new_df = new_df[['open', 'high', 'low', 'close', 'volume']]
    new_df['datetime'] =  pd.date_range(start='9:30', end='16:00', periods=len(new_df))
    new_df = new_df.set_index('datetime')
    plt.figure(figsize=(20, 10))
    mpl.plot(new_df, type='candle', volume=True)
    plt.show()
    return None



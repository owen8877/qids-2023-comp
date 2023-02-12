from pandas import DataFrame
import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_volume(df: DataFrame, asset: int, day: int):
    """create a dataframe that takes index's asset and day equal to asset and day"""
    new_df = df[(df.index.get_level_values('asset') == asset) & (df.index.get_level_values('day') == day)]
    new_df = new_df[['open', 'high', 'low', 'close', 'volume']]
    new_df['datetime'] = pd.date_range(start='9:30', end='16:00', periods=len(new_df))
    new_df = new_df.set_index('datetime')
    plt.figure(figsize=(20, 10))
    mpf.plot(new_df, type='candle', volume=True)
    plt.show()
    return None



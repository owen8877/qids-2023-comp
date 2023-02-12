- extract intraday-market features
  - mean price
  - volatility (i.e. intraday standard deviation)
  - mean volume
  - simple version signature: (DataFrame: single-indexed (timeslot,)) -> DataFrame
- visualize market intraday and get some idea
  - implement a simple visualization function that takes in the market dataframe and index of (asset, day)
  - https://github.com/matplotlib/mplfinance
- backtest framework
  - build on existing draft
    1. feature extraction function: pre-processed df -> feature df, caching option?
    2. align different features and apply training model
    3. for each validation fold, compute metrics
  - make sure backtesting (validation) and test submission use the same procedure
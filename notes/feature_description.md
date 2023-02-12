
## Overall setting:
- 54 assets/investments
- 50 timeslots a day for Market data
- 1-min frequency for Market data?
- 1000 days
## Features in Fundamental data:
- "date_time": sAdB format
- "turnoveRatio": Turnover Ratio
- "transactionAmount": total amount of transactions (多少手)
- "pe_ttm": Price-to-Earnings Ratio (Trailing Twelve Months)
- "pe": Price-to-Earnings Ratio
- "pb": Price-to-Book Ratio
- "ps": Price-to-Sales Ratio
- "pcf": Price-to-Cash-Flow Ratio

## Features in Market data:
- 'date_time': sAdBpC format
- 'open': price at the beginning of this timeslot
- 'close': price at the end of this timeslot
- 'high':  highest price in this timeslot
- 'low':  lowest price in this timeslot
- 'volume':  total amount of units traded （多少股）
- 'money:  total amount of money traded

## Features in Return data:
- 'date_time': sAdB format
- 'return': label data, return of investment
    - Two-day fixed period holding;
    - Trade at the end of the day;
    - Can't use data after, e.g. can't obtain return of sAd(B-1) as it needs sAd(B+1)
    - Computed percentage: $$ sAdB = \frac{sAd(B+2)p50 - sAdBp50}{sAdBp50}$$






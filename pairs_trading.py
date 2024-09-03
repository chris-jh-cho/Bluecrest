# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import yfinance as yf


from utils.call_data import get_sp500_tickers, get_nasdaq100_tickers
from utils.ts_methods import cointegration_test, calculate_zscore
from utils.ts_methods import calculate_pairwise_rolling_correlation
from utils.ts_methods import filter_high_correlation_pairs
from utils.ts_methods import backtest_pairs_trading
from utils.ts_methods import calculate_beta
from utils.plotting_tool import plot_spreads_in_zscores 
from utils.plotting_tool import plot_correlation_results


"""
--------------------------------------------------------------------------------
Task 1:
Using Yahoo Finance, or another source of your choosing, download end of day 
data for the components of the S&P 500, Russell 2000, and/or Nasdaq 100 
(as well as the performance of the relevant index/indices) to a local database.
--------------------------------------------------------------------------------
"""


"""
Step 1a: Pull data
"""

sp500_tickers = get_sp500_tickers()
nasdaq100_tickers = get_nasdaq100_tickers()
russell2000_tickers = pd.read_csv("russell_2000_components.csv").Ticker.to_list()

combined_set = set(sp500_tickers + russell2000_tickers + nasdaq100_tickers)
stock_tickers = list(combined_set)

index_tickers = ["^GSPC", "^NDX", "^RUT"]

start_date = '2000-01-01'
end_date = '2024-08-31'

# Only retain the "Adj Close" column
stock_data = yf.download(stock_tickers, start_date, end_date)['Adj Close']
index_data = yf.download(index_tickers, start_date, end_date)['Adj Close']

stock_data.to_csv('stock_data.csv')
index_data.to_csv('index_data.csv')


"""
Step 1b: Clean data
"""

stock_data = pd.read_csv('stock_data.csv', index_col=0)
index_data = pd.read_csv('index_data.csv', index_col=0)

# concat the index series at the end of single stock data
stock_data = pd.concat([stock_data, index_data], axis=1)

# drop empty columns
stock_data = stock_data.dropna(axis=1, how='all')

# drop if any of the most recent month's data is missing
last_month = stock_data.tail(22)
missing_columns = last_month.columns[last_month.isnull().any()]
stock_data = stock_data.drop(missing_columns, axis=1)

# drop if more than 5% of the data is missing oveer the past 3 years
three_years_ago = datetime.date.today() - datetime.timedelta(days = 3*252)
stock_data_past_3_years = stock_data[pd.to_datetime(stock_data.index).date 
                                     >= three_years_ago]
missing_percentage = (stock_data_past_3_years.isnull().sum() /
                       len(stock_data_past_3_years))
columns_to_drop = missing_percentage[missing_percentage > .05].index
stock_data = stock_data.drop(columns_to_drop, axis=1)

# convert index from datetime to date (helps with plotting)
stock_data.index = pd.to_datetime(stock_data.index).date


"""
--------------------------------------------------------------------------------
Task 2:
From the collected securities, identify the most highly correlated pairs 
(using correlation, mean reversion speeds, etc.). Consider ways to control for 
noisy correlation structure (OLS/Kalman/Cov matrix smoothing/etc..).
--------------------------------------------------------------------------------
"""


"""
Step 2a: Identify most highly correlated pairs across entire time series
"""

returns_data = np.log(stock_data/stock_data.shift(1))
correlation_matrix = returns_data.corr()

# select pairs with high correlation (0.8 chosen arbitrarily)
high_corr_pairs = []
threshold = 0.8

for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > threshold:
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                    correlation_matrix.columns[j]))

# amongst identified pairs, run cointegration test with 5% significance level
cointegrated_pairs = []
for pair in high_corr_pairs:
    p_value = cointegration_test(stock_data, pair[0], pair[1])[1]
    if p_value < 0.05:
        cointegrated_pairs.append(pair)

# take a look
plot_spreads_in_zscores(stock_data, cointegrated_pairs)


"""
Step 2b: Identify most highly correlated pairs in recent times
---
Unsurprisingly, there does not exist a pair that has been mean-reverting
since 2000. Therefore, we now look at more recent data
"""

# Restrict stock universe to save computational cost
train_data = stock_data.iloc[-1010:-252, -100:]

# Calculate 3-months rolling correlation
results = calculate_pairwise_rolling_correlation(train_data.diff(), window=63)

# Select correlated pairs based on final smoothed observation with corr > 0.8
filtered_pairs = filter_high_correlation_pairs(results, threshold=0.8, 
                                               method='ema', 
                                               smoothing_param=63)

# plot the rolling average of each selected pair
# N.B. the parameters of Kalman Smoothing hasn't been tuned, and the algorithm
# takes a while to run
plot_correlation_results(results, filtered_pairs)



"""
--------------------------------------------------------------------------------
Task 3:
Given the list of correlated pairs from the recent step, backtest a mean 
reverting trading strategy based on various entry/exit conditions, trade 
duration, or any other relevant parameters.
--------------------------------------------------------------------------------
"""


"""
Step 3: Run backtest on the selected pairs using training and test data.
---
Moving window z-score thresholds are used for the purpose of capturing mean
reversion. A trade is entered if the spread hits the entry threshold is hit, 
and exited if the spread mean-reverts back to the exit threshold or hits the 
stoploss threshold.
---
Note that the rolling correlation window to select pairs in section 2 is 
independent from the window in this function.
---
Settings can be modifieid to play around with thresholds and the lookback
duration of the window. There is also a choice of smoothing between standard
rolling window z-score and exponentially weighted z-score.
"""

# Look at performance based on training data (4 years ago to 1 year ago)
results_train, series_train = backtest_pairs_trading(train_data, filtered_pairs,
                                                     window=63, 
                                                     entry_threshold=2, 
                                                     exit_threshold=0, 
                                                     stoploss_threshold=4,
                                                     smoothing='rolling')


# test data is data from 1 year ago
test_data = stock_data.iloc[-252:, -100:]

# Look at performance based on test data (4 years ago to 1 year ago)
results_test, series_test = backtest_pairs_trading(test_data, filtered_pairs, 
                                                   window=63, entry_threshold=2,
                                                   exit_threshold=0, 
                                                   stoploss_threshold=4,
                                                   smoothing='rolling')



"""
--------------------------------------------------------------------------------
Task 4:
Provide summary statistics and relevant analysis to help evaluate strategy 
performance. Evaluate the sensitivity of analysis to various parameters.
--------------------------------------------------------------------------------
"""


"""
Step 4: Repeat Step 3 with different rolling window and thresolds.
---
The total returns, annualised returns, annualised volatility, information ratio,
market beta vs S&P 500, and maximum drawdowns are provided as part of strategy
evaluation below.
---
Reducing the window makes strategy more volatile, as thresholds are breached
more frequenty. This leads to amplied total returns in both directions, and an
inconclusive effect on risk-adjusted returns.
---
Unsurprisingly, changing the entry threshold has a similar effect as changing
the window. The wider the threshold, lower the volatility and vice-versa.
The stoploss threshold exists to reduce the drawdown of the straategy, and
pushing this out further leave the strategy susceptible to tail risk.
---
Changing the smoothing method between rolling mean/std to an exponentially
weighted mean/std also gave inconclusive results
---
An interesting idea for analysis would be to calculate z-scores of the spread 
using an smoothed lookback window that expands. This will allow the strategy to 
encompass the history, benefitting from long run patterns, but also staying 
adaptable to recent information.
"""

# Look at performance based on test data
results_test, series_test = backtest_pairs_trading(test_data, filtered_pairs, 
                                                   window=126, entry_threshold=1,
                                                   exit_threshold=0, 
                                                   stoploss_threshold=3,
                                                   smoothing='ema')

# take a look at basic properties such as returns, volatility and info ratio
print(results_test)


# market returns
spx_returns = test_data['^GSPC']/test_data['^GSPC'][0] - 1

# pick a strategy
strategy_returns = series_test['Returns series'][6].dropna()

# calculate market beta
beta = calculate_beta(strategy_returns, spx_returns)
print(beta)

# calculate maximum drawdown
min(strategy_returns)


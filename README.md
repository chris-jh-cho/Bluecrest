# Python Coding Assessment: Equity Pair Trading Backtester

---

**Getting Started:**

Please check the Python version in runtime.txt and the package requirements in requirements.txt. 

Solutions to tasks 1 to 4 can be found in the Python script named pairs_trading.py. 

The solution to task 5 can be found in the Jupyter notebook named pairs_trading.ipynb.

---

**Task 1:**

Using Yahoo Finance, or another source of your choosing, download end of day data for the components of the S&P 500, Russell 2000, and/or Nasdaq 100 (as well as the performance of the relevant index/indices) to a local database.
Make sure to pull the indices



**Task 2:**

From the collected securities, identify the most highly correlated pairs (using correlation, mean reversion speeds, etc.). Consider ways to control for noisy correlation structure (OLS/Kalman/Cov matrix smoothing/etc..).
OLS to smooth
Change trading horizon to smooth
EMA to smooth
Kalman smooth (maybe particle filter if series isn’t stationary?)
Heatmap of correlation matrix that evolves with time/tenor



**Task 3:**

Given the list of correlated pairs from the recent step, backtest a mean reverting trading strategy based on various entry/exit conditions, trade duration, or any other relevant parameters.
Standard deviation
Drawdown limit
Rules determined upon trade entry (probably not dynamically adjusted)



**Task 4:**

Provide summary statistics and relevant analysis to help evaluate strategy performance. Evaluate the sensitivity of analysis to various parameters.
Test against last 2 years



**Task 5:**

Provide a simple interactive solution for quick experimentation of a given pair strategy with a range of parameter choices. This is probably easiest to do with a Jupyter notebook and Streamlit, but Dash/Holoviz Panel/etc also work fine.
Tenor
Smoothing method
Stop loss/take profit method



**Task 6:**

Package the above in a self-contained Github repository that should be straightforward to download and get running from scratch on either a Windows or a Mac
Note: It is important to note that we are less interested in the outright return characteristics of the strategy, and more interested in the overall structure of the solution and design choices behind it.


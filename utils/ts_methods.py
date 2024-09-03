# import packages
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from itertools import combinations


def cointegration_test(stock_data, stock1, stock2):
    
    """
    Performs a cointegration test between two stock time series.

    Parameters:
    - stock_data: DataFrame containing the historical price data of multiple 
                  stocks. Each column should represent the price series of one 
                  stock.
    - stock1: String representing the name of the first stock.
    - stock2: String representing the name of the second stock.

    Returns:
    - result: A tuple containing the output of the Augmented Dickey-Fuller (ADF) test 
              applied to the residuals of the linear regression between the two stocks. 
    """

    clean_data = pd.concat([stock_data[stock1], 
                            stock_data[stock2]], axis=1).dropna()
    model = sm.OLS(clean_data[stock1], sm.add_constant(clean_data[stock2]))
    result = model.fit()
    return sm.tsa.adfuller(result.resid)


def calculate_zscore(spread):
    return (spread - spread.mean()) / spread.std()


def apply_ema(series, span):
    
    """
    Apply Exponentially Weighted Moving Average (EMA) to a series.
    
    Parameters:
    - series: The pandas Series to which EMA will be applied.
    - span: The span parameter for the EMA.
    
    Returns:
    - A pandas Series with the EMA applied.
    """
    
    return series.ewm(span=span, adjust=False).mean()


def apply_kalman_filter(series):
    
    """
    Apply Kalman Filter to a time series using the filterpy library.
    
    Parameters:
    - series: The pandas Series to which Kalman Filter will be applied.
    
    Returns:
    - A pandas Series with the Kalman Filter applied.
    """
    
    # Drop all missing data - this should have been cleaned out beforehand
    series = series.dropna()
    
    # Initialize KalmanFilter object
    kf = KalmanFilter(dim_x=1, dim_z=1)
    
    # Define initial state (just the first observation, no velocity)
    kf.x = np.array([[series.iloc[0]]])
    
    # Define the state transition matrix (identity matrix for no change in state)
    kf.F = np.array([[1]])
    
    # Define the measurement function (identity matrix)
    kf.H = np.array([[1]])
    
    # Define the initial covariance matrix for the state (initial uncertainty)
    kf.P *= 10.0  # Start with high initial uncertainty
    
    # Define the measurement noise (covariance of measurement errors)
    kf.R = 0.1  # Measurement noise, relatively small for correlation values
    
    # Define the process noise covariance (how much we expect the state to change)
    kf.Q = np.array([[0.1]])  # Small process noise for smooth changes
    
    # Storage for the filtered results
    filtered_state_means = []

    for measurement in series:
        kf.predict()
        kf.update(measurement)
        filtered_state_means.append(kf.x[0, 0])

    return pd.Series(filtered_state_means, index=series.index)


def calculate_pairwise_rolling_correlation(stock_data, window=252):
    
    """
    Calculate the rolling pairwise correlation for each pair of stocks in the 
    DataFrame.
    
    Parameters:
    - stock_data: A pandas DataFrame where each column represents a stock's 
                  time-series data.
    - window: The rolling window size for calculating correlations.
    
    Returns:
    - A dictionary where each key is a tuple representing a stock pair, 
      and the value is the rolling correlation series.
    """
    
    results = {}

    # Loop through all unique pairs of stocks in the DataFrame
    for stock_A, stock_B in combinations(stock_data.columns, 2):
        # Calculate rolling correlation between the two stocks
        rolling_corr = stock_data[stock_A].rolling(window=window).corr(stock_data[stock_B])
        
        # Store the rolling correlation series in the results dictionary
        results[(stock_A, stock_B)] = rolling_corr
    
    return results


def filter_high_correlation_pairs(results, threshold=0.8, method='last', 
                                  smoothing_param=None):
    
    """
    Filter out stock pairs with high correlations using different methods.
    
    Parameters:
    - results: A dictionary where each key is a tuple representing a stock pair,
               and the value is the rolling correlation series.
    - threshold: The correlation threshold above which pairs are considered 
                 highly correlated.
    - method: The method to use ('last', 'ema', 'kalman').
    - smoothing_param: The parameter for the smoothing method (e.g., span for 
                       EMA).
    
    Returns:
    - A list of stock pairs that have had high correlations.
    """
    
    filtered_pairs = []

    for pair, rolling_corr in results.items():
        # Apply the chosen method
        if method == 'ema':
            if smoothing_param is None:
                smoothing_param = 63  # Default span is a month if not provided
            smoothed_corr = apply_ema(rolling_corr.dropna(), span=smoothing_param)
        elif method == 'kalman':
            smoothed_corr = apply_kalman_filter(rolling_corr.dropna())
        elif method == 'last':
            smoothed_corr = rolling_corr.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check if the correlation is above the threshold

        if smoothed_corr.dropna().iloc[-1] > threshold:
            filtered_pairs.append(pair)

    return filtered_pairs



def backtest_pairs_trading(stock_data, pairs, window=21, entry_threshold=2.0, 
                           exit_threshold=0.0, stoploss_threshold=3.0, 
                           smoothing="rolling"):
    
    """
    Backtest a pairs trading strategy for mean reversion with stop-loss.
    
    Parameters:
    - stock_data:           DataFrame with historical price data for all stocks.
    - pairs:                Either a single tuple containing a stock pair (e.g., 
                            ('AAPL', 'MSFT')), or a list of tuples containing  
                            stock pairs (e.g., [('AAPL', 'MSFT'), ...]).
    - window:               Rolling window size for z-score calculation.
    - entry_threshold:      Z-score threshold for entering a trade.
    - exit_threshold:       Z-score threshold for exiting a trade.
    - stoploss_threshold:   Z-score threshold for stop-loss.
    - smoothing:            The type of smoothing method to use for mean and  
                            standard deviation calculation. Accepts "rolling" 
                            for rolling mean or "ema" for exponential moving 
                            average (EMA).
                 
    Returns:
    - A DataFrame with the strategy's performance metrics.
    """
    
    # Ensure pairs is a list, even if a single pair is provided
    if isinstance(pairs, tuple):
        pairs = [pairs]
    
    results = {}
    series = {}
    plot_count = len(pairs)
    cols = 2  # Maximum number of charts per row
    rows = (plot_count + cols - 1) // cols  # Calculate the number of rows needed
    
    # Initialize plots for spreads
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    
    # Initialize plots for position
    position_fig, position_axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    position_axes = position_axes.flatten()
    
    # Initialize plots for cumulative P&L
    pnl_fig, pnl_axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    pnl_axes = pnl_axes.flatten()
    
    for idx, (stock_A, stock_B) in enumerate(pairs):
        if stock_A not in stock_data.columns or stock_B not in stock_data.columns:
            continue
        
        # Get price data
        prices_A = stock_data[stock_A]
        prices_B = stock_data[stock_B]
        
        # Calculate spread
        spread = prices_A - prices_B
        
        # Calculate rolling mean or EMA
        if smoothing == 'ema':
            mean = spread.ewm(span=window, adjust=False).mean()
            std = spread.ewm(span=window, adjust=False).std()
        elif smoothing == 'rolling':
            mean = spread.rolling(window=window).mean()
            std = spread.rolling(window=window).std()
        else:
            raise ValueError("Invalid smoothing method. Choose 'rolling' or 'ema'.")
        
        # Convert Z-score thresholds to spread values
        entry_spread_threshold_high = mean + entry_threshold * std
        entry_spread_threshold_low = mean - entry_threshold * std
        exit_spread_threshold = mean
        stoploss_spread_threshold_high = mean + stoploss_threshold * std
        stoploss_spread_threshold_low = mean - stoploss_threshold * std
        
        # Calculate z-score
        z_scores = (spread - mean) / std
        
        # Initialize positions and P&L
        position = pd.Series(0, index=stock_data.index, dtype=float)
        
        # Dummy variable to indicate whether the strategy can enter a position 
        dummy = True
    
        # Variable indicating whether the strategy is currently long/short
        long = True
        
        for i in range(1, len(z_scores)):
            if dummy:
                if z_scores.iloc[i] < -entry_threshold:
                    position.iloc[i] = 1  # Long position
                    long = True
                    dummy = False
                elif z_scores.iloc[i] > entry_threshold:
                    position.iloc[i] = -1  # Short position
                    long = False
                    dummy = False
            else:
                if np.abs(z_scores.iloc[i]) > stoploss_threshold:
                    position.iloc[i] = 0  # Exit position
                    dummy = True
                elif z_scores.iloc[i] > exit_threshold and long:
                    position.iloc[i] = 0  # Exit position
                    dummy = True
                elif z_scores.iloc[i] < exit_threshold and not long:
                    position.iloc[i] = 0  # Exit position
                    dummy = True
                else:
                    position.iloc[i] = position.iloc[i-1]  # Hold the previous position
        
        # Calculate returns
        returns = stock_data[stock_A].pct_change() - stock_data[stock_B].pct_change()
        strategy_returns = position.shift(1) * returns
        
        # Calculate cumulative P&L
        cumulative_pnl = (strategy_returns + 1).cumprod() - 1
        #cumulative_pnl = strategy_returns.cumsum()
        
        # Calculate performance metrics
        total_return = cumulative_pnl.iloc[-1]
        annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
        annualized_volatility = strategy_returns.std() * np.sqrt(252)
        info_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan
        
        # Store results
        results[(stock_A, stock_B)] = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Information Ratio': info_ratio
        }
        
        # Store series
        series[(stock_A, stock_B)] = {
            'Returns series': cumulative_pnl,
            'Position': position
        }
        
        # Plot Spread (instead of Z-Score)
        if idx < len(axes):
            axes[idx].plot(spread, label=f'{stock_A} vs {stock_B} Spread', color='navy', alpha=.7)
            axes[idx].plot(entry_spread_threshold_high, color='hotpink', linestyle='--', label='Entry Threshold', alpha=.7)
            axes[idx].plot(entry_spread_threshold_low, color='hotpink', linestyle='--', alpha=.7)
            axes[idx].plot(exit_spread_threshold, color='cyan', linestyle='--', label='Exit Threshold', alpha=.7)
            axes[idx].plot(stoploss_spread_threshold_high, color='red', linestyle='--', label='Stop-Loss Threshold', alpha=.7)
            axes[idx].plot(stoploss_spread_threshold_low, color='red', linestyle='--', alpha=.7)
            axes[idx].set_title(f'{stock_A} vs {stock_B}')
            axes[idx].set_xlabel('Date')
            axes[idx].set_ylabel('Spread')
            axes[idx].legend()
            
            # Reduce the frequency of x-axis labels
            axes[idx].xaxis.set_major_locator(plt.MaxNLocator(5))  # Display at most 5 x-axis labels
        
        # Plot position
        if idx < len(position_axes):
            position_axes[idx].plot(position, label=f'{stock_A} vs {stock_B} Position', color='mediumpurple')
            position_axes[idx].set_title(f'{stock_A} vs {stock_B}')
            position_axes[idx].set_xlabel('Date')
            position_axes[idx].set_ylabel('Current exposure')
            
            # Reduce the frequency of x-axis labels
            position_axes[idx].xaxis.set_major_locator(plt.MaxNLocator(5))  # Display at most 5 x-axis labels
            
        # Plot P&L
        if idx < len(pnl_axes):
            pnl_axes[idx].plot(cumulative_pnl, label=f'{stock_A} vs {stock_B} P&L', color='cyan')
            pnl_axes[idx].set_title(f'{stock_A} vs {stock_B}')
            pnl_axes[idx].set_xlabel('Date')
            pnl_axes[idx].set_ylabel('Cumulative P&L')
            
            # Reduce the frequency of x-axis labels
            pnl_axes[idx].xaxis.set_major_locator(plt.MaxNLocator(5))  # Display at most 5 x-axis labels
            
    # Hide any unused subplots
    for i in range(len(pairs), len(axes)):
        axes[i].axis('off')
        pnl_axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(results).T, pd.DataFrame(series).T


def calculate_beta(strategy_returns, market_returns):
    """
    Calculate the beta of a trading strategy's returns against the market.

    Parameters:
    - strategy_returns: Series or DataFrame containing the strategy's returns.
    - market_returns: Series or DataFrame containing the market's returns.

    Returns:
    - beta: The beta of the strategy with respect to the market.
    """
    
    # Ensure the returns are aligned and drop any missing data
    data = pd.concat([strategy_returns, market_returns], axis=1).dropna()
    
    # Rename the columns for easier reference
    data.columns = ['Strategy', 'Market']
    
    # Add a constant to the market returns (to account for the intercept in the regression)
    X = sm.add_constant(data['Market'])
    
    # Run OLS regression: Strategy returns ~ Market returns
    model = sm.OLS(data['Strategy'], X)
    results = model.fit()
    
    # The beta is the coefficient of the Market return
    beta = results.params['Market']
    
    return beta

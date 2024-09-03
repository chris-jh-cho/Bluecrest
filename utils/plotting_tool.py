import numpy as np
import math
import matplotlib.pyplot as plt

from utils.ts_methods import calculate_zscore, apply_ema, apply_kalman_filter


def plot_spreads_in_zscores(stock_data, cointegrated_pairs):
    """
    Plots the spread in z-scores for each pair in cointegrated_pairs.
    
    Parameters:
    - stock_data: DataFrame containing stock price data for multiple stocks.
    - cointegrated_pairs: List of tuples containing pairs of stock tickers.
    
    Returns:
    - Displays a grid of plots showing the spread in z-scores for each pair.
    """
    
    if isinstance(cointegrated_pairs, tuple):
        cointegrated_pairs = [cointegrated_pairs]
    
    num_pairs = len(cointegrated_pairs)
    cols = 2  # Fixed number of columns
    rows = (num_pairs // cols) + (num_pairs % cols > 0)  # Calculate the required number of rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))  # Adjust figsize based on the number of rows
    
    # Flatten axes array if there is only one row or column
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if cols == 1:
        axes = np.expand_dims(axes, axis=1)
    
    index = 0
    for i in range(rows):
        for j in range(cols):
            if index >= num_pairs:
                axes[i, j].axis('off')  # Hide any unused subplots
                continue
                
            stock1, stock2 = cointegrated_pairs[index]
            spread = stock_data[stock1] - stock_data[stock2]
            zscore = calculate_zscore(spread)
            
            ax = axes[i, j]
            ax.plot(zscore)
            ax.axhline(2.0, color='red', linestyle='--')
            ax.axhline(-2.0, color='red', linestyle='--')
            ax.axhline(0, color='black', linestyle='-')
            ax.set_title(f"{stock1} - {stock2}")
            ax.set_xticks([])  # Remove x-axis ticks for better layout
            
            index += 1
            
    # Adjust layout and display the plot
    fig.suptitle("Spread (in z-scores) evolution of Cointegrated Pairs", fontsize=14)
    plt.tight_layout()
    plt.show()
    
    
    
def plot_correlation_results(results, filtered_pairs, ema_span=30):
    """
    Plot a grid of line charts showing the rolling correlations for the selected pairs,
    including EMA and Kalman smoothed lines.

    Parameters:
    - results: A dictionary where each key is a tuple representing a stock pair, and the value is the rolling correlation series.
    - filtered_pairs: A list of stock pairs that were filtered based on high recent correlations.
    - ema_span: The span for EMA smoothing.

    Returns:
    - A plot displaying the rolling correlations, EMA, and Kalman smoothed lines for the filtered stock pairs.
    """
    # Set the number of columns
    cols = 2
    # Dynamically calculate the number of rows needed
    rows = math.ceil(len(filtered_pairs) / cols)

    # Initialize the plot with the calculated dimensions
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    # Plot each pair
    for i, pair in enumerate(filtered_pairs):
        if pair in results:
            rolling_corr = results[pair].dropna()

            # Apply smoothing
            ema_smoothed = apply_ema(rolling_corr, span=ema_span)
            kalman_smoothed = apply_kalman_filter(rolling_corr)

            # Plot original rolling correlation
            axes[i].plot(rolling_corr, label='Rolling Correlation', color='black', alpha=0.5)
            
            # Plot EMA smoothed
            axes[i].plot(ema_smoothed, label=f'EMA (span={ema_span})', color='hotpink', alpha=0.5)
            
            # Plot Kalman smoothed
            axes[i].plot(kalman_smoothed, label='Kalman Smoothed', color='cyan', alpha=0.5)

            # Set titles and labels
            axes[i].set_title(f'{pair[0]} vs {pair[1]}')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Correlation')
            axes[i].legend()
            axes[i].grid(True)

    # Hide any remaining empty subplots
    for i in range(len(filtered_pairs), rows * cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
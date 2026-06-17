import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

def log_returns(prices):
    """
    Compute log returns.
    """
    return np.log(prices/prices.shift(1)).dropna()

def historical_volatility(returns, window=21):
    return (returns.rolling(window).std()*np.sqrt(252)) #21-day realized volatility

def plot_historical_volatility(hv):
    plt.figure(figsize=(10, 5))
    plt.plot(
        hv.index,
        hv.values
    )
    plt.title("21-Day Historical Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid()
    plt.show()

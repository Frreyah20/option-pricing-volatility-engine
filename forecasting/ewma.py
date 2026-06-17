import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def ewma_volatility(returns, lam=0.94):
    returns = returns.dropna()
    variance = np.zeros(len(returns))
    variance[0] = float(returns.var())
    for t in range(1, len(returns)):
        variance[t] = (lam * variance[t-1] + (1-lam) * returns.iloc[t-1]**2)
    ewma_vol = np.sqrt(variance) * np.sqrt(252)
    return pd.Series(ewma_vol, index=returns.index)

def plot_ewma_volatility(ewma_vol):
    plt.figure(figsize=(10, 5))
    plt.plot(ewma_vol.index, ewma_vol.values)
    plt.title("EWMA Volatility Forecast")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid()
    plt.show()


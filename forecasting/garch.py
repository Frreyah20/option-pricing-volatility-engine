from arch import arch_model 
import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 

def fit_garch(returns):
    """
    Fit GARCH(1,1) model.
    """
    returns = returns.dropna()
    model = arch_model(returns * 100, vol="Garch", p = 1, q = 1, mean = "Zero")
    results = model.fit(disp="off")
    return results

def garch_volatility(results):
    vol = (results.conditional_volatility/100 * np.sqrt(252))
    return pd.Series(vol, index=results.conditional_volatility.index) 

def plot_garch_volatility(garch_vol):
    plt.figure(figsize=(10, 5))
    plt.plot(garch_vol.index, garch_vol.values)
    plt.title("GARCH(1, 1) Volatility")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid()
    plt.show()

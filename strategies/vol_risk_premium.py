import pandas as pd 
import matplotlib.pyplot as plt 


def compute_vrp(iv_series, rv_series):
    df = pd.concat([iv_series, rv_series], axis = 1)
    df.columns = ["iv", "rv"]
    df["vrp"] = (df["iv"] - df["rv"])
    return df 

def plot_vrp(df):
    plt.figure(figsize=(10, 15))
    plt.plot(df.index, df["vrp"])
    plt.axhline(0, linestyle="--")
    plt.title("Volatility Risk Premium")
    plt.ylabel("IV - RV")
    plt.grid()
    plt.show()
    
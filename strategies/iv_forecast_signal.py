import pandas as pd
import matplotlib.pyplot as plt 

def compute_signal(vrp_df):
    df = vrp_df.copy()
    df["signal"] = 0
    df.loc[df["vrp"] > 0.05, "signal"] = -1
    df.loc[df["vrp"] < -0.05, "signal"] = 1
    return df

def plot_signal(df):
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["vrp"], label="VRP")
    plt.axhline(0.05,linestyle="--",color="red")
    plt.axhline(-0.05,linestyle="--",color="green")
    plt.title("Volatility Signal")
    plt.legend()
    plt.grid()
    plt.show()
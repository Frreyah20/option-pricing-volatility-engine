import pandas as pd
import matplotlib.pyplot as plt 

def compute_iv_forecast_spread(iv, forecast):
    common_index = iv.index.intersection(forecast.index)
    iv = iv.loc[common_index]
    forecast = forecast.loc[common_index]
    df = pd.DataFrame({"iv": iv, "forecast":forecast})
    df["spread"] = (df["iv"] - df["forecast"])
    return df 

def generate_signal(spread_df, threshold=0.03):
    df = spread_df.copy()
    df["signal"] = 0
    df.loc[df["spread"] > threshold, "signal"] = -1
    df.loc[df["spread"] < -threshold, "signal"] = 1
    return df

def plot_spread(df):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["spread"], label = "IV-Forecast")
    plt.axhline(-0.03, linestyle="--")
    plt.title("IV vs Forecast Vol Spread")
    plt.legend()
    plt.grid()
    plt.show()


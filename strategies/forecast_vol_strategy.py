import pandas as pd
import matplotlib.pyplot as plt


def compute_forecast_spread(iv_series, forecast_vol):
    df = pd.DataFrame({"iv": iv_series,"forecast_vol": forecast_vol})
    df["spread"] = df["iv"] - df["forecast_vol"]
    return df
    
def generate_forecast_signal(spread_df,threshold=0.03):
    spread_df["signal"] = 0
    spread_df.loc[spread_df["spread"] > threshold, "signal"] = -1
    spread_df.loc[spread_df["spread"] < -threshold, "signal"] = 1
    return spread_df

def plot_forecast_spread(spread_df):
    plt.figure(figsize=(10,5))
    plt.plot(spread_df.index,spread_df["spread"],label="IV - Forecast Vol")
    plt.axhline(0,linestyle="--")
    plt.axhline(0.03,linestyle="--")
    plt.axhline(-0.03,linestyle="--")
    plt.title("Forecast Vol Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid()
    plt.show()
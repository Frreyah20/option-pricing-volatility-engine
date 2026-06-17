import pandas as pd
import matplotlib.pyplot as plt 


def run_backtest(signal_df, returns):
    df = signal_df.copy()
    df["asset_return"] = returns
    df["position"] = df["signal"].shift(1).fillna(0)
    df["turnover"] = df["position"].diff().abs().fillna(0)
    df["transaction_cost"] = df["turnover"] * 0.0005
    df["strategy_return"] = df["position"] * df["asset_return"] - df["transaction_cost"]
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    print(
    "Total Cost =",
    round(df["transaction_cost"].sum(), 4)
    )
    return df

def plot_equity_curve(backtest_df):
    plt.figure(figsize=(10,5))
    plt.plot(backtest_df.index, backtest_df["equity_curve"])
    plt.title("Volatility Strategy Equity Curve")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.show()
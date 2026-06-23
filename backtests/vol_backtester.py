import pandas as pd
import matplotlib.pyplot as plt 


def run_backtest(signal_df, returns, txn_cost_bps=5, slippage_bps=0):
    df = signal_df.copy()
    df["asset_return"] = returns
    df["position"] = df["signal"].shift(1).fillna(0)
    df["turnover"] = df["position"].diff().abs().fillna(0)
    
    # Calculate transaction costs and slippage separately
    df["transaction_cost"] = df["turnover"] * (txn_cost_bps / 10000)
    df["slippage_cost"] = df["turnover"] * (slippage_bps / 10000)
    df["total_cost"] = df["transaction_cost"] + df["slippage_cost"]
    
    # Compute strategy returns net of all execution costs
    df["strategy_return"] = df["position"] * df["asset_return"] - df["total_cost"]
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    
    # Reporting Metrics
    total_turnover = df["turnover"].sum()
    num_trades = int((df["turnover"] > 0).sum())
    total_txn_cost = df["transaction_cost"].sum()
    total_slippage = df["slippage_cost"].sum()
    total_exec_cost = df["total_cost"].sum()
    cost_drag_pct = total_exec_cost * 100
    
    print("\n--- Execution Costs Report ---")
    print(f"Total Turnover: {total_turnover:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Transaction Costs: {total_txn_cost:.4f}")
    print(f"Slippage Costs: {total_slippage:.4f}")
    print(f"Total Execution Costs: {total_exec_cost:.4f}")
    print(f"Cost Drag (%): {cost_drag_pct:.2f}%")
    print("------------------------------")
    
    return df

def plot_equity_curve(backtest_df):
    plt.figure(figsize=(10,5))
    plt.plot(backtest_df.index, backtest_df["equity_curve"])
    plt.title("Volatility Strategy Equity Curve")
    plt.ylabel("Portfolio Value")
    plt.grid()
    plt.show()
import pandas as pd 
from strategies.forecast_vol_strategy import generate_forecast_signal
from backtests.vol_backtester import run_backtest
from backtests.performance import sharpe_ratio, max_drawdown, win_rate, annual_return

def threshold_sweep(spread_df, returns, thresholds):
    results = []
    for threshold in thresholds:
        temp_df = spread_df.copy()
        temp_df = generate_forecast_signal(temp_df, threshold=threshold)
        backtest_df = run_backtest(temp_df, returns)
        results.append(
            {
                "Threshold": threshold,
                "Sharpe": sharpe_ratio(backtest_df["strategy_return"]),
                "MaxDD": max_drawdown(backtest_df["equity_curve"]),
                "WinRate": win_rate(backtest_df["strategy_return"]),
                "AnnualReturns": annual_return(backtest_df["equity_curve"])

            }
        )
    return pd.DataFrame(results) 
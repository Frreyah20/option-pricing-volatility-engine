import pandas as pd
from backtests.performance import sharpe_ratio, max_drawdown, win_rate, annual_return

def analyze_regime_performance(backtest_df,regime_series):
    df = backtest_df.copy()
    df["regime"] = regime_series.reindex(df.index)
    results = []
    for regime in ["LOW", "MEDIUM", "HIGH"]:
        subset = df[df["regime"] == regime]
        if len(subset) == 0:
            continue
        results.append({
            "Regime": regime,
            "Sharpe":
                sharpe_ratio(
                    subset["strategy_return"]
                ),

            "MaxDD":
                max_drawdown(
                    subset["equity_curve"]
                ),

            "WinRate":
                win_rate(
                    subset["strategy_return"]
                ),

            "AnnualReturn":
                annual_return(
                    subset["equity_curve"]
                )
        })

    results_df = pd.DataFrame(results)
    return results_df
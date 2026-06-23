import yfinance as yf
import pandas as pd

from forecasting.ml_features import build_ml_dataset
from forecasting.walk_forward import walk_forward_xgb
from strategies.forecast_vol_strategy import compute_forecast_spread, generate_forecast_signal
from backtests.vol_backtester import run_backtest
from backtests.performance import sharpe_ratio, max_drawdown, win_rate, annual_return

def main():
    # 1. Load data
    prices = yf.download("SPY", start="2020-01-01", auto_adjust=True)["Close"]
    prices = prices.squeeze()
    returns = prices.pct_change().dropna()
    
    # 2. Build ML Dataset
    df = build_ml_dataset(prices)
    
    # 3. Generate Walk-Forward Forecasts
    wf_forecast = walk_forward_xgb(df)
    
    # 4. Generate Signal
    iv_proxy = df["future_vol"]
    common_index = iv_proxy.index.intersection(wf_forecast.index)
    iv_aligned = iv_proxy.loc[common_index]
    forecast_aligned = wf_forecast.loc[common_index]
    
    spread_df = compute_forecast_spread(iv_aligned, forecast_aligned)
    spread_df = generate_forecast_signal(spread_df, threshold=0.01)
    
    # 5. Cost Sensitivity Experiment
    slippage_scenarios = [0, 5, 10, 20]
    results = []
    
    print("\nStarting Cost-Sensitivity Experiment...")
    for slip_bps in slippage_scenarios:
        # Run the backtest with a static 5 bps txn cost, iterating over slippage scenarios
        backtest_df = run_backtest(spread_df, returns, txn_cost_bps=5, slippage_bps=slip_bps)
        
        # Calculate performance metrics
        sr = sharpe_ratio(backtest_df["strategy_return"])
        ar = annual_return(backtest_df["equity_curve"])
        md = max_drawdown(backtest_df["equity_curve"])
        wr = win_rate(backtest_df["strategy_return"])
        total_exec_cost = backtest_df["total_cost"].sum()
        
        results.append({
            "Slippage (bps)": slip_bps,
            "Sharpe Ratio": round(sr, 3),
            "Annual Return": f"{round(ar * 100, 2)}%",
            "Max Drawdown": f"{round(md * 100, 2)}%",
            "Win Rate": f"{round(wr * 100, 2)}%",
            "Total Execution Costs": round(total_exec_cost, 4)
        })
        
    # 6. Output formatted comparison table
    results_df = pd.DataFrame(results)
    print("\n===== COST SENSITIVITY ANALYSIS =====")
    print(results_df.to_string(index=False))
    print("=====================================")

if __name__ == "__main__":
    main()

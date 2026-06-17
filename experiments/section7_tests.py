import yfinance as yf
from forecasting.historical import (log_returns, historical_volatility)
from forecasting.garch import (fit_garch, garch_volatility)
from strategies.vol_risk_premium import (compute_vrp, plot_vrp)
from strategies.iv_forecast_signal import compute_signal, plot_signal
from backtests.vol_backtester import run_backtest, plot_equity_curve
from backtests.performance import sharpe_ratio, max_drawdown, win_rate, annual_return
from forecasting.garch import fit_garch, garch_volatility
from strategies.forecast_vol_strategy import compute_forecast_spread, generate_forecast_signal, plot_forecast_spread
import numpy as np 
import pandas as pd 

print("\n===== VRP STRATEGY =====")
prices = yf.download("SPY", start="2020-01-01", auto_adjust=True)["Close"]
returns = log_returns(prices)
returns = returns.squeeze()
rv = historical_volatility(returns, window=21)
results = fit_garch(returns)
iv_proxy = rv.shift(-21)
vrp_df = compute_vrp(iv_proxy, rv)
print(vrp_df.tail())
plot_vrp(vrp_df)
signal_df = compute_signal(vrp_df)
print(signal_df[["vrp","signal"]].tail())
plot_signal(signal_df)
backtest_df = run_backtest(signal_df, returns)
print(backtest_df[["signal","position","strategy_return","equity_curve"]].tail())
plot_equity_curve(backtest_df)
sharpe = sharpe_ratio(backtest_df["strategy_return"])
mdd = max_drawdown(backtest_df["equity_curve"]) 
wr = win_rate(backtest_df["strategy_return"])
ann_ret = annual_return(backtest_df["equity_curve"])
print()
print("Sharpe      =", round(sharpe, 3))
print("Max DD      =", round(mdd, 3))
print("Win Rate    =", round(wr, 3))
print("Annual Ret  =", round(ann_ret, 3))

print("\n===== GARCH FORECAST STRATEGY =====")
#returns = np.log(prices / prices.shift(1)).dropna()
#results = fit_garch(returns)
forecast_vol = garch_volatility(results)
common_index = iv_proxy.index.intersection(forecast_vol.index)
iv_aligned = iv_proxy.loc[common_index]
forecast_aligned = forecast_vol.loc[common_index]
print(iv_aligned.tail())
print(forecast_aligned.tail())
spread_df = compute_forecast_spread(iv_aligned,forecast_aligned)
print(spread_df.tail())
spread_df = spread_df.dropna()
spread_df = generate_forecast_signal(spread_df,threshold=0.03)
print(spread_df[["spread", "signal"]].tail())
plot_forecast_spread(spread_df)
backtest_df = run_backtest(spread_df, returns)
plot_equity_curve(backtest_df)
print("Sharpe =",sharpe_ratio(backtest_df["strategy_return"]))
print("Max DD =",max_drawdown(backtest_df["equity_curve"]))
print("Win Rate =",win_rate(backtest_df["strategy_return"]))
print("Annual Return =",annual_return(backtest_df["equity_curve"]))



import yfinance as yf
import pandas as pd

from forecasting.ml_features import (
    build_ml_dataset,
    train_random_forest,
    train_xgboost,
    get_rf_forecast_series,
    get_xgb_forecast_series
)

from strategies.forecast_vol_strategy import (
    compute_forecast_spread,
    generate_forecast_signal,
    plot_forecast_spread
)

from backtests.vol_backtester import (
    run_backtest,
    plot_equity_curve
)

from backtests.performance import (
    sharpe_ratio,
    max_drawdown,
    annual_return,
    win_rate
)

from strategies.iv_series import build_iv_proxy
from strategies.iv_forecast_spread import compute_iv_forecast_spread, generate_signal, plot_spread 
from experiments.threshold_sweep import threshold_sweep
# ==================================================
# Load Data
# ==================================================

prices = yf.download(
    "SPY",
    start="2020-01-01",
    auto_adjust=True
)["Close"]

if isinstance(prices, pd.DataFrame):
    prices = prices.squeeze()

# ==================================================
# Build ML Dataset
# ==================================================

df = build_ml_dataset(prices)

print(df.head())

# ==================================================
# Train Models
# ==================================================

rf_model, rf_preds, y_test = train_random_forest(df)

xgb_model, xgb_preds, _ = train_xgboost(df)

returns = prices.pct_change().dropna()
rv = returns.rolling(21).std()*(252**0.5)
iv_proxy = build_iv_proxy(rv)

# ==================================================
# Full Forecast Series
# ==================================================

rf_forecast = get_rf_forecast_series(
    rf_model,
    df
)

xgb_forecast = get_xgb_forecast_series(
    xgb_model,
    df
)

# ==================================================
# RANDOM FOREST STRATEGY
# ==================================================

print("\n===== RANDOM FOREST STRATEGY =====")

rf_spread = compute_forecast_spread(
    df["future_vol"],
    rf_forecast
)

rf_spread = generate_forecast_signal(
    rf_spread,
    threshold=0.03
)

print(
    rf_spread[
        ["spread", "signal"]
    ].tail()
)

plot_forecast_spread(rf_spread)

rf_backtest = run_backtest(
    rf_spread,
    df["ret_1d"]
)

plot_equity_curve(rf_backtest)

rf_sharpe = sharpe_ratio(
    rf_backtest["strategy_return"]
)

rf_mdd = max_drawdown(
    rf_backtest["equity_curve"]
)

rf_wr = win_rate(
    rf_backtest["strategy_return"]
)

rf_ann = annual_return(
    rf_backtest["equity_curve"]
)

print("Sharpe =", round(rf_sharpe, 3))
print("Max DD =", round(rf_mdd, 3))
print("Win Rate =", round(rf_wr, 3))
print("Annual Return =", round(rf_ann, 3))

# ==================================================
# XGBOOST STRATEGY
# ==================================================

print("\n===== XGBOOST STRATEGY =====")

xgb_spread = compute_forecast_spread(
    df["future_vol"],
    xgb_forecast
)

xgb_spread = generate_forecast_signal(
    xgb_spread,
    threshold=0.03
)

print(
    xgb_spread[
        ["spread", "signal"]
    ].tail()
)

plot_forecast_spread(xgb_spread)

xgb_backtest = run_backtest(
    xgb_spread,
    df["ret_1d"]
)

plot_equity_curve(xgb_backtest)

xgb_sharpe = sharpe_ratio(
    xgb_backtest["strategy_return"]
)

xgb_mdd = max_drawdown(
    xgb_backtest["equity_curve"]
)

xgb_wr = win_rate(
    xgb_backtest["strategy_return"]
)

xgb_ann = annual_return(
    xgb_backtest["equity_curve"]
)

print("Sharpe =", round(xgb_sharpe, 3))
print("Max DD =", round(xgb_mdd, 3))
print("Win Rate =", round(xgb_wr, 3))
print("Annual Return =", round(xgb_ann, 3))

# ==================================================
# MODEL COMPARISON
# ==================================================

results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost"
    ],
    "Sharpe": [
        rf_sharpe,
        xgb_sharpe
    ],
    "MaxDD": [
        rf_mdd,
        xgb_mdd
    ],
    "WinRate": [
        rf_wr,
        xgb_wr
    ],
    "AnnualReturn": [
        rf_ann,
        xgb_ann
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results)

# ==================================================
# RANDOM FOREST IV
# ==================================================

print("\n===== RANDOM FOREST IV =====")

rf_spread_iv = compute_iv_forecast_spread(
    iv_proxy,
    rf_forecast
)

rf_spread_iv = generate_signal(
    rf_spread_iv,
    threshold=0.03
)

plot_spread(rf_spread_iv)

rf_backtest_iv = run_backtest(
    rf_spread_iv,
    returns
)

plot_equity_curve(rf_backtest_iv)

rf_sharpe_iv = sharpe_ratio(
    rf_backtest_iv["strategy_return"]
)

rf_mdd_iv = max_drawdown(
    rf_backtest_iv["equity_curve"]
)

rf_wr_iv = win_rate(
    rf_backtest_iv["strategy_return"]
)

rf_ann_iv = annual_return(
    rf_backtest_iv["equity_curve"]
)

print("Sharpe =", round(rf_sharpe_iv, 3))
print("Max DD =", round(rf_mdd_iv, 3))
print("Win Rate =", round(rf_wr_iv, 3))
print("Annual Return =", round(rf_ann_iv, 3))

# ==================================================
# XGBOOST IV
# ==================================================

print("\n===== XGBOOST IV =====")

xgb_spread_iv = compute_iv_forecast_spread(
    iv_proxy,
    xgb_forecast
)

xgb_spread_iv = generate_signal(
    xgb_spread_iv,
    threshold=0.03
)

plot_spread(xgb_spread_iv)

xgb_backtest_iv = run_backtest(
    xgb_spread_iv,
    returns
)

plot_equity_curve(xgb_backtest_iv)

xgb_sharpe_iv = sharpe_ratio(
    xgb_backtest_iv["strategy_return"]
)

xgb_mdd_iv = max_drawdown(
    xgb_backtest_iv["equity_curve"]
)

xgb_wr_iv = win_rate(
    xgb_backtest_iv["strategy_return"]
)

xgb_ann_iv = annual_return(
    xgb_backtest_iv["equity_curve"]
)

print("Sharpe =", round(xgb_sharpe_iv, 3))
print("Max DD =", round(xgb_mdd_iv, 3))
print("Win Rate =", round(xgb_wr_iv, 3))
print("Annual Return =", round(xgb_ann_iv, 3))

# ==================================================
# ENSEMBLE FOREST
# ==================================================

ensemble_forecast = 0.4 * rf_forecast + 0.4 * xgb_forecast + 0.2 * df["garch_vol"]
ensemble_spread = compute_iv_forecast_spread(iv_proxy, ensemble_forecast)
ensemble_spread = generate_signal(ensemble_spread, threshold = 0.03)
ensemble_backtest = run_backtest(ensemble_spread, df["ret_1d"])

ensemble_sharpe_iv = sharpe_ratio(
    ensemble_backtest["strategy_return"]
)

ensemble_mdd_iv = max_drawdown(
    ensemble_backtest["equity_curve"]
)

ensemble_wr_iv = win_rate(
    ensemble_backtest["strategy_return"]
)

ensemble_ann_iv = annual_return(
    ensemble_backtest["equity_curve"]
)

print("\n===== ENSEMBLE IV =====")
print("Sharpe =", round(ensemble_sharpe_iv, 3))
print("Max DD =", round(ensemble_mdd_iv, 3))
print("Win Rate =", round(ensemble_wr_iv, 3))
print("Annual Return =", round(ensemble_ann_iv, 3))

# ==================================================
# THRESHOLD SWEEP
# ==================================================

thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
rf_sweep = threshold_sweep(rf_spread_iv, returns,thresholds)
xgb_sweep = threshold_sweep(xgb_spread_iv, returns,thresholds)
ensemble_sweep = threshold_sweep(ensemble_spread, returns,thresholds)
print("\n===== RF THRESHOLD SWEEP =====")
print(rf_sweep.sort_values("Sharpe", ascending=False))

print("\n===== XGB THRESHOLD SWEEP =====")
print(xgb_sweep.sort_values("Sharpe", ascending=False))

print("\n===== ENSEMBLE THRESHOLD SWEEP =====")
print(ensemble_sweep.sort_values("Sharpe", ascending=False))



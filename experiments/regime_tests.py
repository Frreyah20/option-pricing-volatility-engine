import yfinance as yf

from forecasting.historical import (
    log_returns,
    historical_volatility
)

from forecasting.garch import (
    fit_garch,
    garch_volatility
)

from strategies.regime_analysis import (
    classify_regimes,
    plot_regimes
)

from strategies.forecast_vol_strategy import (
    compute_forecast_spread,
    generate_forecast_signal
)

from strategies.regime_performance import (
    analyze_regime_performance
)

from backtests.vol_backtester import (
    run_backtest
)

from backtests.performance import (
    sharpe_ratio,
    max_drawdown
)

from strategies.regime_filter import apply_regime_filter

from strategies.regime_transitions import (
    identify_transitions
)

from strategies.regime_transitions import (
    transition_performance
)

from backtests.vol_backtester import plot_equity_curve

# ======================================
# Load Data
# ======================================

prices = yf.download(
    "SPY",
    start="2020-01-01",
    auto_adjust=True
)["Close"]

prices = prices.squeeze()

returns = log_returns(prices)

rv = historical_volatility(
    returns,
    window=21
)

# ======================================
# Regime Classification
# ======================================

regime = classify_regimes(rv)

print(regime.value_counts())

print()
print(regime.head())

print()
print(regime.tail())

plot_regimes(rv, regime)

# ======================================
# Build Forecast Vol Strategy
# ======================================

results = fit_garch(returns)

forecast_vol = garch_volatility(results)

iv_proxy = rv.shift(-21)

common_index = iv_proxy.index.intersection(
    forecast_vol.index
)

iv_aligned = iv_proxy.loc[common_index]

forecast_aligned = forecast_vol.loc[common_index]

spread_df = compute_forecast_spread(
    iv_aligned,
    forecast_aligned
)

spread_df = spread_df.dropna()

spread_df = generate_forecast_signal(
    spread_df,
    threshold=0.02
)

# ======================================
# Backtest
# ======================================

backtest_df = run_backtest(
    spread_df,
    returns
)

# ======================================
# Regime Performance Analysis
# ======================================

regime_results = analyze_regime_performance(
    backtest_df,
    regime
)

print()
print("===== REGIME PERFORMANCE =====")
print(regime_results)

low_df = apply_regime_filter(
    spread_df,
    regime,
    ["LOW"]
)

low_backtest = run_backtest(
    low_df,
    returns
)

medium_df = apply_regime_filter(
    spread_df,
    regime,
    ["MEDIUM"]
)

medium_backtest = run_backtest(
    medium_df,
    returns
)

high_df = apply_regime_filter(
    spread_df,
    regime,
    ["HIGH"]
)

high_backtest = run_backtest(
    high_df,
    returns
)

print("\n===== LOW REGIME =====")
print("Sharpe =", sharpe_ratio(low_backtest["strategy_return"]))
print("Max DD =", max_drawdown(low_backtest["equity_curve"]))

print("\n===== MEDIUM REGIME =====")
print("Sharpe =", sharpe_ratio(medium_backtest["strategy_return"]))
print("Max DD =", max_drawdown(medium_backtest["equity_curve"]))

print("\n===== HIGH REGIME =====")
print("Sharpe =", sharpe_ratio(high_backtest["strategy_return"]))
print("Max DD =", max_drawdown(high_backtest["equity_curve"]))

regime = classify_regimes(rv)
transition_df = identify_transitions(regime)
print()
print("===== TRANSITIONS =====")
print(
    transition_df["transition"]
    .value_counts()
)

perf = transition_performance(
    transition_df,
    returns,
    horizon=5
)

print()
print("===== TRANSITION PERFORMANCE =====")
print(perf)

print("\n===== LOW + MEDIUM FILTER =====")

filtered_df = apply_regime_filter(
    spread_df,
    regime,
    ["LOW", "MEDIUM"]
)

filtered_backtest = run_backtest(
    filtered_df,
    returns
)

print(
    "Original Sharpe =",
    sharpe_ratio(backtest_df["strategy_return"])
)

print(
    "Filtered Sharpe =",
    sharpe_ratio(filtered_backtest["strategy_return"])
)

print(
    "Original Max DD =",
    max_drawdown(backtest_df["equity_curve"])
)

print(
    "Filtered Max DD =",
    max_drawdown(filtered_backtest["equity_curve"])
)

plot_equity_curve(backtest_df)
plot_equity_curve(filtered_backtest)
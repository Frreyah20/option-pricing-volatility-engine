import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from forecasting.historical import (
    log_returns,
    historical_volatility
)

from forecasting.garch import (
    fit_garch,
    garch_volatility
)

from strategies.regime_analysis import (
    classify_regimes
)

from strategies.forecast_vol_strategy import (
    compute_forecast_spread,
    generate_forecast_signal
)

from strategies.regime_performance import (
    analyze_regime_performance
)

from strategies.regime_filter import (
    apply_regime_filter
)

from strategies.regime_transitions import (
    identify_transitions,
    transition_performance
)

from backtests.vol_backtester import (
    run_backtest
)

from backtests.performance import (
    sharpe_ratio,
    max_drawdown
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Regime Analysis",
    layout="wide"
)

st.title("Volatility Regime Analysis")

# =====================================
# LOAD DATA
# =====================================

with st.spinner("Loading market data..."):

    prices = yf.download(
        "SPY",
        start="2020-01-01",
        auto_adjust=True
    )["Close"]

    prices = prices.squeeze()

    returns = log_returns(prices)

# =====================================
# REGIMES
# =====================================

rv = historical_volatility(
    returns,
    window=21
)

regime = classify_regimes(rv)

# =====================================
# REGIME DISTRIBUTION
# =====================================

st.header("Regime Distribution")

counts = regime.value_counts()

fig, ax = plt.subplots(figsize=(8,4))

ax.bar(
    counts.index,
    counts.values
)

ax.set_ylabel("Observations")
ax.set_title("Volatility Regimes")

st.pyplot(fig)

st.dataframe(
    counts.rename("Count"),
    use_container_width=True
)

# =====================================
# BUILD STRATEGY
# =====================================

results = fit_garch(
    returns
)

forecast_vol = garch_volatility(
    results
)

iv_proxy = rv.shift(-21)

common_index = (
    iv_proxy.index.intersection(
        forecast_vol.index
    )
)

iv_aligned = iv_proxy.loc[
    common_index
]

forecast_aligned = forecast_vol.loc[
    common_index
]

spread_df = compute_forecast_spread(
    iv_aligned,
    forecast_aligned
)

spread_df = spread_df.dropna()

spread_df = generate_forecast_signal(
    spread_df,
    threshold=0.02
)

backtest_df = run_backtest(
    spread_df,
    returns
)

# =====================================
# REGIME PERFORMANCE
# =====================================

st.header("Regime Performance")

regime_results = analyze_regime_performance(
    backtest_df,
    regime
)

st.dataframe(
    regime_results,
    use_container_width=True
)

# =====================================
# FILTERED STRATEGY
# =====================================

st.header("Low + Medium Regime Filter")

filtered_df = apply_regime_filter(
    spread_df,
    regime,
    ["LOW", "MEDIUM"]
)

filtered_backtest = run_backtest(
    filtered_df,
    returns
)

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Original Sharpe",
        round(
            sharpe_ratio(
                backtest_df["strategy_return"]
            ),
            3
        )
    )

    st.metric(
        "Original Max DD",
        round(
            max_drawdown(
                backtest_df["equity_curve"]
            ),
            3
        )
    )

with col2:

    st.metric(
        "Filtered Sharpe",
        round(
            sharpe_ratio(
                filtered_backtest["strategy_return"]
            ),
            3
        )
    )

    st.metric(
        "Filtered Max DD",
        round(
            max_drawdown(
                filtered_backtest["equity_curve"]
            ),
            3
        )
    )

# =====================================
# EQUITY CURVE COMPARISON
# =====================================

st.header("Equity Curve Comparison")

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    backtest_df.index,
    backtest_df["equity_curve"],
    label="Original"
)

ax.plot(
    filtered_backtest.index,
    filtered_backtest["equity_curve"],
    label="Filtered"
)

ax.legend()

ax.set_title(
    "Original vs Regime Filtered Strategy"
)

ax.grid(True)

st.pyplot(fig)

# =====================================
# REGIME TRANSITIONS
# =====================================

st.header("Regime Transitions")

transition_df = identify_transitions(
    regime
)

transition_counts = (
    transition_df["transition"]
    .value_counts()
    .reset_index()
)

transition_counts.columns = [
    "Transition",
    "Count"
]

st.dataframe(
    transition_counts,
    use_container_width=True
)

# =====================================
# TRANSITION PERFORMANCE
# =====================================

st.header("Transition Performance")

transition_perf = transition_performance(
    transition_df,
    returns,
    horizon=5
)

st.dataframe(
    transition_perf,
    use_container_width=True
)
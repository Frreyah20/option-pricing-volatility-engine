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

from strategies.vol_risk_premium import (
    compute_vrp
)

from strategies.iv_forecast_signal import (
    compute_signal
)

from strategies.forecast_vol_strategy import (
    compute_forecast_spread,
    generate_forecast_signal
)

from backtests.vol_backtester import (
    run_backtest
)

from backtests.performance import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    annual_return
)

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Trading Strategies",
    layout="wide"
)

st.title("Volatility Trading Strategies")

# =====================================
# DATA
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
# VRP STRATEGY
# =====================================

st.header("Volatility Risk Premium Strategy")

rv = historical_volatility(
    returns,
    window=21
)

iv_proxy = rv.shift(-21)

vrp_df = compute_vrp(
    iv_proxy,
    rv
)

vrp_df = compute_signal(vrp_df)

vrp_backtest = run_backtest(
    vrp_df,
    returns
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sharpe",
    round(
        sharpe_ratio(
            vrp_backtest["strategy_return"]
        ),
        3
    )
)

col2.metric(
    "Max DD",
    round(
        max_drawdown(
            vrp_backtest["equity_curve"]
        ),
        3
    )
)

col3.metric(
    "Win Rate",
    round(
        win_rate(
            vrp_backtest["strategy_return"]
        ),
        3
    )
)

col4.metric(
    "Annual Return",
    round(
        annual_return(
            vrp_backtest["equity_curve"]
        ),
        3
    )
)

# =====================================
# VRP CHART
# =====================================

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    vrp_df.index,
    vrp_df["vrp"]
)

ax.axhline(
    0.05,
    linestyle="--"
)

ax.axhline(
    -0.05,
    linestyle="--"
)

ax.set_title(
    "Volatility Risk Premium"
)

ax.grid(True)

st.pyplot(fig)

# =====================================
# VRP EQUITY CURVE
# =====================================

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    vrp_backtest.index,
    vrp_backtest["equity_curve"]
)

ax.set_title(
    "VRP Strategy Equity Curve"
)

ax.grid(True)

st.pyplot(fig)

# =====================================
# FORECAST SPREAD STRATEGY
# =====================================

st.header("GARCH Forecast Spread Strategy")

results = fit_garch(
    returns
)

forecast_vol = garch_volatility(
    results
)

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

spread_backtest = run_backtest(
    spread_df,
    returns
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sharpe",
    round(
        sharpe_ratio(
            spread_backtest["strategy_return"]
        ),
        3
    )
)

col2.metric(
    "Max DD",
    round(
        max_drawdown(
            spread_backtest["equity_curve"]
        ),
        3
    )
)

col3.metric(
    "Win Rate",
    round(
        win_rate(
            spread_backtest["strategy_return"]
        ),
        3
    )
)

col4.metric(
    "Annual Return",
    round(
        annual_return(
            spread_backtest["equity_curve"]
        ),
        3
    )
)

# =====================================
# SPREAD CHART
# =====================================

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    spread_df.index,
    spread_df["spread"]
)

ax.axhline(
    0.02,
    linestyle="--"
)

ax.axhline(
    -0.02,
    linestyle="--"
)

ax.set_title(
    "Forecast Spread"
)

ax.grid(True)

st.pyplot(fig)

# =====================================
# EQUITY CURVE
# =====================================

fig, ax = plt.subplots(
    figsize=(10,5)
)

ax.plot(
    spread_backtest.index,
    spread_backtest["equity_curve"]
)

ax.set_title(
    "Forecast Spread Strategy Equity Curve"
)

ax.grid(True)

st.pyplot(fig)

# =====================================
# STRATEGY COMPARISON
# =====================================

st.header("Strategy Comparison")

comparison_df = pd.DataFrame(
    {
        "Strategy": [
            "VRP",
            "Forecast Spread"
        ],

        "Sharpe": [
            sharpe_ratio(
                vrp_backtest["strategy_return"]
            ),

            sharpe_ratio(
                spread_backtest["strategy_return"]
            )
        ],

        "MaxDD": [
            max_drawdown(
                vrp_backtest["equity_curve"]
            ),

            max_drawdown(
                spread_backtest["equity_curve"]
            )
        ],

        "WinRate": [
            win_rate(
                vrp_backtest["strategy_return"]
            ),

            win_rate(
                spread_backtest["strategy_return"]
            )
        ],

        "AnnualReturn": [
            annual_return(
                vrp_backtest["equity_curve"]
            ),

            annual_return(
                spread_backtest["equity_curve"]
            )
        ]
    }
)

st.dataframe(
    comparison_df,
    use_container_width=True
)
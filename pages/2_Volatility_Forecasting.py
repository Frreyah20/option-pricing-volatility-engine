import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from forecasting.historical import (
    log_returns,
    historical_volatility
)

from forecasting.ewma import (
    ewma_volatility
)

from forecasting.garch import (
    fit_garch,
    garch_volatility
)

# =====================================
# Page Config
# =====================================

st.set_page_config(
    page_title="Volatility Forecasting",
    layout="wide"
)

st.title("Volatility Forecasting")

st.markdown(
    """
    Compare Historical Volatility,
    EWMA Forecasts, and GARCH Forecasts.
    """
)

# =====================================
# Sidebar Inputs
# =====================================

ticker = st.sidebar.text_input(
    "Ticker",
    value="SPY"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime(2020, 1, 1)
)

# =====================================
# Load Data
# =====================================

with st.spinner("Downloading market data..."):

    prices = yf.download(
        ticker,
        start=start_date,
        auto_adjust=True
    )["Close"]

    prices = prices.squeeze()

    returns = log_returns(prices)

# =====================================
# Forecast Models
# =====================================

hv = historical_volatility(
    returns,
    window=21
)

ewma = ewma_volatility(
    returns
)

results = fit_garch(
    returns
)

garch = garch_volatility(
    results
)

# =====================================
# Metrics
# =====================================

st.subheader("Latest Forecasts")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Historical Vol",
        f"{hv.iloc[-1]:.2%}"
    )

with col2:
    st.metric(
        "EWMA Vol",
        f"{ewma.iloc[-1]:.2%}"
    )

with col3:
    st.metric(
        "GARCH Vol",
        f"{garch.iloc[-1]:.2%}"
    )

# =====================================
# Forecast Chart
# =====================================

st.subheader("Volatility Forecast Comparison")

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    hv.index,
    hv.values,
    label="Historical Volatility"
)

ax.plot(
    ewma.index,
    ewma.values,
    label="EWMA Forecast"
)

ax.plot(
    garch.index,
    garch.values,
    label="GARCH Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel("Annualized Volatility")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# =====================================
# Model Comparison Table
# =====================================

st.subheader("Current Model Comparison")

comparison_df = pd.DataFrame(
    {
        "Model": [
            "Historical Volatility",
            "EWMA",
            "GARCH"
        ],
        "Current Forecast": [
            hv.iloc[-1],
            ewma.iloc[-1],
            garch.iloc[-1]
        ]
    }
)

st.dataframe(
    comparison_df,
    use_container_width=True
)

# =====================================
# Summary Statistics
# =====================================

st.subheader("Summary Statistics")

stats_df = pd.DataFrame(
    {
        "Model": [
            "Historical Volatility",
            "EWMA",
            "GARCH"
        ],
        "Mean": [
            hv.mean(),
            ewma.mean(),
            garch.mean()
        ],
        "Maximum": [
            hv.max(),
            ewma.max(),
            garch.max()
        ],
        "Minimum": [
            hv.min(),
            ewma.min(),
            garch.min()
        ]
    }
)

st.dataframe(
    stats_df,
    use_container_width=True
)
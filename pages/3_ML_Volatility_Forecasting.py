import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from forecasting.ml_features import (
    build_ml_dataset,
    train_random_forest,
    train_xgboost,
    compare_models
)

st.set_page_config(
    page_title="ML Volatility Forecasting",
    layout="wide"
)

st.title("Machine Learning Volatility Forecasting")

st.markdown(
    """
    Compare Random Forest and XGBoost models
    for forecasting future volatility.
    """
)

# ======================================
# Sidebar
# ======================================

ticker = st.sidebar.text_input(
    "Ticker",
    value="SPY"
)

# ======================================
# Cached Functions
# ======================================

@st.cache_data
def load_prices(ticker):
    prices = yf.download(
        ticker,
        start="2020-01-01",
        auto_adjust=True
    )["Close"]

    return prices


@st.cache_data
def build_dataset(ticker):
    prices = load_prices(ticker)
    return build_ml_dataset(prices)


@st.cache_resource
def train_models(ticker):

    df = build_dataset(ticker)

    rf_model, rf_preds, y_test = train_random_forest(df)

    xgb_model, xgb_preds, _ = train_xgboost(df)

    return (
        df,
        rf_model,
        rf_preds,
        xgb_model,
        xgb_preds,
        y_test
    )

# ======================================
# Load Everything
# ======================================

with st.spinner("Training models..."):

    (
        df,
        rf_model,
        rf_preds,
        xgb_model,
        xgb_preds,
        y_test
    ) = train_models(ticker)

# ======================================
# Performance Comparison
# ======================================

st.subheader("Model Performance")

results = compare_models(
    y_test,
    rf_preds,
    xgb_preds
)

st.dataframe(
    results,
    use_container_width=True
)

# ======================================
# Prediction Plot
# ======================================

st.subheader("Forecast Comparison")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    y_test.values,
    label="Actual Future Vol"
)

ax.plot(
    rf_preds,
    label="Random Forest"
)

ax.plot(
    xgb_preds,
    label="XGBoost"
)

ax.set_xlabel("Test Observations")
ax.set_ylabel("Volatility")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ======================================
# Feature Importance
# ======================================

feature_names = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "hist_vol",
    "ewma_vol",
    "garch_vol"
]

st.subheader("Random Forest Feature Importance")

rf_importance = pd.Series(
    rf_model.feature_importances_,
    index=feature_names
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))

rf_importance.plot(
    kind="barh",
    ax=ax
)

st.pyplot(fig)

# ======================================
# XGBoost Importance
# ======================================

st.subheader("XGBoost Feature Importance")

xgb_importance = pd.Series(
    xgb_model.feature_importances_,
    index=feature_names
).sort_values()

fig, ax = plt.subplots(figsize=(8, 4))

xgb_importance.plot(
    kind="barh",
    ax=ax
)

st.pyplot(fig)

# ======================================
# Best Model
# ======================================

best_model = results.sort_values(
    "RMSE"
).iloc[0]["Model"]

st.success(
    f"Best model based on RMSE: {best_model}"
)
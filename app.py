import streamlit as st

st.set_page_config(
    page_title="Volatility Research Platform",
    layout="wide"
)

st.title("Volatility Forecasting & Trading Research Platform")

st.markdown("""
This platform combines:

- Black-Scholes Pricing
- Greeks Analytics
- Implied Volatility
- Volatility Forecasting
- Machine Learning Models
- Trading Strategies
- Regime Analysis

Use the navigation menu on the left to explore the platform.
""")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pricing Models", "3")

with col2:
    st.metric("ML Models", "2")

with col3:
    st.metric("Strategy Sharpe", "2.29") 
import streamlit as st

from pricing.black_scholes import black_scholes_call
from pricing.greeks import (
    call_delta,
    gamma,
    vega,
    call_theta,
    call_rho
)

st.title("Option Analytics")

S = st.sidebar.number_input(
    "Spot Price",
    value=100.0
)

K = st.sidebar.number_input(
    "Strike",
    value=100.0
)

T = st.sidebar.number_input(
    "Time to Maturity",
    value=1.0
)

r = st.sidebar.number_input(
    "Risk Free Rate",
    value=0.05
)

sigma = st.sidebar.number_input(
    "Volatility",
    value=0.20
)

price = black_scholes_call(
    S,
    K,
    T,
    r,
    sigma
)

delta = call_delta(
    S,
    K,
    T,
    r,
    sigma
)

g = gamma(
    S,
    K,
    T,
    r,
    sigma
)

v = vega(
    S,
    K,
    T,
    r,
    sigma
)

theta = call_theta(
    S,
    K,
    T,
    r,
    sigma
)

rho = call_rho(
    S,
    K,
    T,
    r,
    sigma
)

st.subheader("Black-Scholes Price")

st.metric(
    "Call Price",
    round(price, 4)
)

st.subheader("Greeks")

col1, col2, col3 = st.columns(3)

col1.metric("Delta", round(delta, 4))
col2.metric("Gamma", round(g, 4))
col3.metric("Vega", round(v, 4))

col1, col2 = st.columns(2)

col1.metric("Theta", round(theta, 4))
col2.metric("Rho", round(rho, 4))
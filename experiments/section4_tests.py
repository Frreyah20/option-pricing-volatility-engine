from data.market_data import get_option_chain, get_spot_price, get_expiries
from research.smile_analysis import compute_smile, plot_smile
from research.skew_analysis import analyze_skew
from research.term_structure import compute_term_structure, plot_term_structure

calls, puts = get_option_chain("SPY", "2026-09-18")
import pandas as pd

S = get_spot_price("SPY")
calls = calls[
    (calls["strike"] >= 0.8 * S) & (calls["strike"] <= 1.2 * S)
]
T = 0.25
r = 0.05

smile_df = compute_smile(calls, S, T, r)
if not smile_df.empty:
    plot_smile(smile_df, S)
    print(smile_df.shape)
    analyze_skew(smile_df, S)

expiries = get_expiries("SPY")

term_df = compute_term_structure("SPY", expiries[:15], S, 0.05)
if not term_df.empty:
    print(term_df)
    plot_term_structure(term_df)


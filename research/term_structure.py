import pandas as pd
from volatility.implied_volatility_solver import implied_vol_bisection
from data.market_data import get_option_chain
from datetime import datetime
import matplotlib.pyplot as plt

def compute_term_structure(ticker, expiries, S, r):
    expiry_list = []
    atm_iv_list = []
    for expiry in expiries:
        calls, puts = get_option_chain(ticker, expiry)
        calls = calls[
            (calls["bid"] > 0)
            & (calls["ask"] > 0)
            & (calls["openInterest"] > 0)
        ]
        if len(calls) == 0:
            continue
        atm_idx = (
            calls["strike"] - S
        ).abs().idxmin()
        atm_row = calls.loc[atm_idx]
        market_price = (
            atm_row["bid"]
            + atm_row["ask"]
        )/2
        K = atm_row["strike"]
        today = datetime.today()
        expiry_date = datetime.strptime(expiry,"%Y-%m-%d")
        days = (expiry_date - today).days
        T = days / 365
        if T <= 0:
            continue
        try:
            iv = implied_vol_bisection(market_price, S, K, T, r)
            if iv < 0.01 or iv > 2.0:
                continue
            expiry_list.append(expiry)
            atm_iv_list.append(iv)
        except Exception:
            continue
    term_df = pd.DataFrame({
        "expiry": expiry_list,
        "days_to_expiry": days_list,
        "atm_iv": atm_iv_list
    })
    if term_df.empty:
        print("No valid term structure data found.")
    return term_df
    
def plot_term_structure(term_df):
    plt.figure(figsize=(8,5))
    plt.plot(term_df["expiry"],term_df["atm_iv"],marker="o")
    plt.xticks(rotation=45)
    plt.xlabel("Expiry")
    plt.ylabel("ATM Implied Volatility")
    plt.title("Volatility Term Structure")
    plt.grid()
    plt.tight_layout()
    plt.show()

        
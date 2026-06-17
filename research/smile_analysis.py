
from volatility.implied_volatility_solver import implied_vol_bisection
import pandas as pd
import matplotlib.pyplot as plt 


def compute_smile(calls, S, T, r):
    strikes = []
    implied_vols = []
    for _, row in calls.iterrows():
        if(row["bid"] <= 0 or row["ask"] <= 0 or row["openInterest"] <= 0):
            continue
        K = row["strike"]
        market_price = (row["bid"] + row["ask"])/2 #mid price
        try:
            iv = implied_vol_bisection(market_price, S, K, T, r)
            if iv< 0.01 or iv > 2.0:
                continue
            strikes.append(K)
            implied_vols.append(iv)
        except Exception:
            continue
    smile_df = pd.DataFrame({
        "strike": strikes,
        "implied_vol": implied_vols
    })
    if smile_df.empty:
        raise ValueError("No valid contracts found.")
    return smile_df

def plot_smile(smile_df, S):
    plt.figure(figsize=(8, 5))
    plt.plot(
        smile_df["strike"],
        smile_df["implied_vol"],
        marker = "o" 
    )
    plt.axvline(x=S, linestyle="--", label = f"Spot = {S:.2f}")
    plt.legend()
    plt.xlabel("Strike")
    plt.ylabel("Implied Volatility")
    plt.title("Volatility Smile")
    plt.grid()
    plt.show()

    
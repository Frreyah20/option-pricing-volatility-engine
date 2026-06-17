import pandas as pd 
from data.market_data import get_option_chain
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 


def build_surface(ticker, expiries, S):
    rows = []
    for expiry in expiries:
        calls, puts = get_option_chain(ticker, expiry)
        for _, row in calls.iterrows():
            iv = row["impliedVolatility"]
            if(pd.isna(iv) or iv < 0.01 or iv > 2.0):
                continue
            spot_lower = 0.8 * S
            spot_upper = 1.2 * S
            if(row["strike"] < spot_lower or row["strike"] > spot_upper):
                continue
            if(row["bid"] <= 0 or row["ask"] <= 0 or row["openInterest"] < 20):
                continue
            rows.append(
                {
                    "expiry": expiry,
                    "strike": row["strike"],
                    "iv": iv
                }
            )
    surface_df = pd.DataFrame(rows)
    return surface_df

def plot_vol_heatmap(heatmap_data):
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heatmap_data, aspect="auto", origin="lower")
    plt.colorbar(im, label="Implied Volatility")
    plt.title("Interpolated Volatility Surface Heatmap")
    plt.xlabel("Strike")
    plt.ylabel("Expiry")
    plt.yticks(
        range(len(heatmap_data.index)),
        heatmap_data.index
    )
    plt.xticks(
        range(0, len(heatmap_data.columns), 20),
        [f"{x:.0f}" for x in heatmap_data.columns[::20]],
        rotation = 45
    )
    plt.tight_layout()
    plt.show()

def interplote_surface(surface_df):
    surface_df = surface_df[
    (surface_df["iv"] >= 0.05)
    & (surface_df["iv"] <= 1.0)
    ]
    heatmap_data = surface_df.pivot_table(
        index="expiry",
        columns="strike",
        values="iv"
    )
    heatmap_data = heatmap_data.sort_index()
    heatmap_data = heatmap_data.sort_index(axis=1)
    heatmap_data = heatmap_data.interpolate(axis =1, limit_direction="both")
    return heatmap_data

def plot_vol_surface(heatmap_data):
    X = np.arange(len(heatmap_data.columns))
    Y = np.arange(len(heatmap_data.index))
    X, Y = np.meshgrid(X, Y)
    Z = heatmap_data.values
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Expiry")
    ax.set_zlabel("Implied Volatility")
    strike_positions = np.arange(
        0, 
        len(heatmap_data.columns),
        max(1, len(heatmap_data.columns)//8)
    )
    ax.set_xticks(strike_positions)
    ax.set_xticklabels(
        [
            f"{heatmap_data.columns[i]:.0f}"
            for i in strike_positions
        ],
        rotation=30
    )
    expiry_positions = np.arange(len(heatmap_data.index))
    ax.set_yticks(expiry_positions)
    ax.set_yticklabels(heatmap_data.index, fontsize=8)
    fig.colorbar(surface, shrink=0.6, aspect=15, label="IV")
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.10, top=0.95)
    plt.show()
    

    
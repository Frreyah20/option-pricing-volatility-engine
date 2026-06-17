import pandas as pd 
import matplotlib.pyplot as plt 

def compare_vols(hist_vol, ewma_vol, garch_vol):
    comparison_df = pd.DataFrame({
        "Model": [
            "Historical Vol",
            "EWMA",
            "GARCH"
        ],
        "Volatility": [
            hist_vol, 
            ewma_vol,
            garch_vol,
        ]
    })
    return comparison_df

def plot_comparison(comparison_df):
    plt.figure(figsize=(8, 5))
    plt.bar(
        comparison_df["Model"],
        comparison_df["Volatility"]
    )
    plt.ylabel("Volatility")
    plt.title("Forecast vs Implied Volatility")
    plt.grid(axis="y")
    plt.show()


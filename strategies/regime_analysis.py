import pandas as pd
import matplotlib.pyplot as plt


def classify_regimes(rv):
    low_threshold = rv.quantile(0.33)
    high_threshold = rv.quantile(0.67)
    regime = pd.Series(index=rv.index, dtype="object")
    regime[rv <= low_threshold] = "LOW"
    regime[(rv > low_threshold) & (rv < high_threshold)] = "MEDIUM"
    regime[rv >= high_threshold] = "HIGH"
    return regime

def plot_regimes(rv, regime):
    plt.figure(figsize=(12,6))
    plt.plot(rv.index, rv.values,label="Realized Volatility")
    low_mask = regime == "LOW"
    med_mask = regime == "MEDIUM"
    high_mask = regime == "HIGH"
    plt.scatter(rv.index[low_mask],rv[low_mask],s=8,label="LOW")
    plt.scatter(rv.index[med_mask], rv[med_mask], s=8, label="MEDIUM")
    plt.scatter(rv.index[high_mask], rv[high_mask], s=8, label="HIGH")
    plt.title("Volatility Regime Classification")
    plt.xlabel("Date")
    plt.ylabel("Realized Volatility")
    plt.legend()
    plt.grid()
    plt.show()
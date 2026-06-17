import numpy as np
import matplotlib.pyplot as plt
from pricing.greeks import call_delta, gamma, vega, call_theta

def plot_delta_vs_spot(K, T, r, sigma):

    spot_prices = np.linspace(50, 150, 200)
    delta_values = []

    for S in spot_prices: 
        delta_values.append(call_delta(S, K, T, r, sigma))

    plt.figure(figsize=(8, 5))
    plt.plot(spot_prices, delta_values)
    plt.axvline(x = K, linestyle = '--', label = f"Strike = {K}")
    plt.legend()
    plt.xlabel("Spot Price")
    plt.ylabel("Call Delta")
    plt.title("Call Delta vs Spot Price")
    plt.grid(True)
    plt.show()

def plot_gamma_vs_spot(K, T, r, sigma):
    spot_prices = np.linspace(50, 150, 200)
    gamma_values = []

    for S in spot_prices: 
        gamma_values.append(gamma(S, K, T, r, sigma))

    plt.figure(figsize=(8, 5))
    plt.plot(spot_prices, gamma_values)
    plt.axvline(x = K, linestyle = '--', label = f"Strike = {K}") 
    plt.legend()
    plt.xlabel("Spot Price")
    plt.ylabel("Call Gamma")
    plt.title("Call Gamma vs Spot Price")
    plt.grid(True)
    plt.show()

def plot_vega_vs_spot(K, T, r, sigma):
    spot_prices = np.linspace(50, 150, 200)
    vega_values = []

    for S in spot_prices: 
        vega_values.append(vega(S, K, T, r, sigma))

    plt.figure(figsize=(8, 5))
    plt.plot(spot_prices, vega_values)
    plt.axvline(x = K, linestyle = '--', label = f"Strike = {K}") 
    plt.legend()
    plt.xlabel("Spot Price")
    plt.ylabel("Call Vega")
    plt.title("Call Vega vs Spot Price")
    plt.grid(True)
    plt.show()

def plot_theta_vs_spot(K, T, r, sigma):
    spot_prices = np.linspace(50, 150, 200)
    theta_values = []

    for S in spot_prices: 
        theta_values.append(call_theta(S, K, T, r, sigma))

    plt.figure(figsize=(8, 5))
    plt.plot(spot_prices, theta_values)
    plt.axvline(x = K, linestyle = '--', label = f"Strike = {K}") 
    plt.legend()
    plt.xlabel("Spot Price")
    plt.ylabel("Call Theta")
    plt.title("Call Theta vs Spot Price")
    plt.grid(True)
    plt.show()
from pricing.greeks import call_delta, put_delta, gamma, vega, call_theta, put_theta, call_rho, put_rho
from pricing.greeks import finitte_difference_delta_call,finite_difference_gamma, finite_difference_vega, finite_difference_rho_call, put_call_parity_check
from visualization.greeks_plots import plot_delta_vs_spot, plot_gamma_vs_spot, plot_vega_vs_spot, plot_theta_vs_spot

S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
h = 0.01

greeks = {
    "Call Delta": call_delta(S, K, T, r, sigma),
    "Put Delta": put_delta(S, K, T, r, sigma),
    "Gamma": gamma(S, K, T, r, sigma),
    "Vega": vega(S, K, T, r, sigma),
    "Call Theta": call_theta(S, K, T, r, sigma),
    "Put Theta": put_theta(S, K, T, r, sigma),
    "Call Rho": call_rho(S, K, T, r, sigma),
    "Put Rho": put_rho(S, K, T, r, sigma)
}

for name, value in greeks.items():
    print(f"{name}: {value:.4f}")

print(f"Analytic Call Delta: {call_delta(S, K, T, r, sigma):.4f}")
print(f"Finite Difference Call Delta: {finitte_difference_delta_call(S, K, T, r, sigma, h):.4f}")
print(f"Analystic Gamma: {gamma(S, K, T, r, sigma):.4f}") 
print(f"Finite Difference Gamma: {finite_difference_gamma(S, K, T, r, sigma, h):.4f}")
print(f"Analytic Vega: {vega(S, K, T, r, sigma):.4f}")
print(f"Finite Difference Vega: {finite_difference_vega(S, K, T, r, sigma, h):.4f}")
print(f"Analytic Call Rho: {call_rho(S, K, T, r, sigma):.4f}")
print(f"Finite Difference Call Rho: {finite_difference_rho_call(S, K, T, r, sigma, h):.4f}")
lhs, rhs = put_call_parity_check(S, K, T, r, sigma)
print(f"C - P = {lhs:.4f}")
print(f"S - K*exp(-rT) = {rhs:.4f}")
print(f"Difference = {abs(lhs-rhs):.10f}")

plot_delta_vs_spot(K, T, r, sigma)
plot_gamma_vs_spot(K, T, r, sigma)
plot_vega_vs_spot(K, T, r, sigma)
plot_theta_vs_spot(K, T, r, sigma)

from pricing.black_scholes import black_scholes_call
from volatility.implied_volatility_solver import implied_vol_bisection, implied_vol_newton
from data.market_data import get_expiries, get_option_chain, get_spot_price
import time


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
h = 0.01 

true_sigma = 0.20
market_price = black_scholes_call(S, K, T, r ,true_sigma)
iv = implied_vol_bisection(market_price, S, K, T, r)
print(f"True Sigma: {true_sigma:.4f}")
print(f"Recovered IV: {iv:.4f}")

test_sigmas = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
for sigma in test_sigmas:
    market_price = black_scholes_call(S, K, T, r, sigma)
    iv = implied_vol_bisection(market_price, S, K, T, r)
    print(f"True Sigma: {sigma:.4f} | Recovered IV: {iv:.4f}")

market_price = black_scholes_call(S, K, T, r ,true_sigma)
iv = implied_vol_newton(market_price, S, K, T, r)
print(f"True Sigma: {true_sigma:.4f}")
print(f"Recovered IV: {iv:.4f}")

test_sigmas = [0.10, 0.15, 0.20, 0.30, 0.40, 0.60]
for sigma in test_sigmas:
    market_price = black_scholes_call(S, K, T, r, sigma)
    iv = implied_vol_newton(market_price, S, K, T, r)
    print(f"True Sigma: {sigma:.4f} | Recovered IV: {iv:.4f}")

market_price = black_scholes_call(S, K, T, r ,sigma)
start = time.perf_counter()
iv_bisection = implied_vol_bisection(market_price, S, K, T, r)
bisection_time = time.perf_counter() - start

start = time.perf_counter()
iv_newton = implied_vol_newton(market_price, S, K, T, r)
newton_time = time.perf_counter() - start

print(
    f"Bisection IV = {iv_bisection:.6f}"
)

print(
    f"Newton IV    = {iv_newton:.6f}"
)

print(
    f"Bisection Time = {bisection_time:.8f}"
)

print(
    f"Newton Time    = {newton_time:.8f}"
)
print("\nMarket Data Retrieval")
print("-" * 50)
ticker = "SPY"
spot = get_spot_price(ticker)
expiries = get_expiries(ticker)
print(f"Spot Price : {spot:.2f}")
print(f"First Expiry : {expiries[0]}")
calls, puts = get_option_chain(ticker,expiries[0])
atm_calls = calls.iloc[
    (calls["strike"] - spot).abs().argsort()[:5]
]

print("\nATM Calls Sample")
print(
    atm_calls[
        [
            "strike",
            "bid",
            "ask",
            "volume",
            "openInterest"
        ]
    ]
)
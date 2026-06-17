from pricing.black_scholes import black_scholes_call
from pricing.greeks import vega


def implied_vol_bisection(market_price, S, K, T, r, tol = 1e-6, max_iteration = 100):
    sigma_low = 0.0001
    sigma_high = 5.0
    
    for _ in range(max_iteration):
        sigma_mid = (sigma_low + sigma_high)/2
        price_mid = black_scholes_call(S, K, T, r, sigma_mid)
        model_price = black_scholes_call(S, K, T, r, sigma_mid)
        if abs(model_price - market_price) < tol :
            return sigma_mid

        if model_price > market_price:
            sigma_high = sigma_mid
        else:
            sigma_low = sigma_mid
    return sigma_mid

def implied_vol_newton(market_price, S, K, T, r, initial_guess = 0.2, tol = 1e-6, max_iteration = 100):
    sigma = initial_guess
    for _ in range(max_iteration):
        model_price = black_scholes_call(S, K, T, r, sigma)
        price_error = model_price - market_price
        if abs(price_error) < tol:
            return sigma
        vega_value = vega(S, K, T, r, sigma)
        sigma = sigma - (price_error/vega_value)
    return sigma
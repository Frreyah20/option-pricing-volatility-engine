import numpy as np
from scipy.stats import norm
from pricing.black_scholes import calculate_d1_d2, black_scholes_call, black_scholes_put

def call_delta(S, K, T, r, sigma):
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1)

def put_delta(S, K, T, r, sigma):
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    gamma_value = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    return gamma_value

def vega(S, K, T, r, sigma):
    d1, _ = calculate_d1_d2(S, K, T, r, sigma)
    vega_value = S*norm.pdf(d1)*np.sqrt(T)
    return vega_value

def call_theta(S, K, T, r, sigma):
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - (r*K*np.exp(-r*T)*norm.cdf(d2))
    return theta/365

def put_theta(S, K, T, r, sigma):
    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)
    theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + (r*K*np.exp(-r*T)*(norm.cdf(-d2)))
    return theta/365

def call_rho(S, K, T, r, sigma):
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    rho = K*T*np.exp(-r*T)*norm.cdf(d2)
    return rho

def put_rho(S, K, T, r, sigma):
    _, d2 = calculate_d1_d2(S, K, T, r, sigma)
    rho = -K*T*np.exp(-r*T)*(norm.cdf(-d2))
    return rho

def finitte_difference_delta_call(S, K, T, r, sigma, h = 0.01):
    price_up = black_scholes_call(S+h, K, T, r, sigma)
    price_down = black_scholes_call(S-h, K, T, r, sigma)
    delta_fd = (price_up - price_down)/(2*h)
    return delta_fd

def finitte_difference_delta_put(S, K, T, r, sigma, h = 0.01):
    price_up = black_scholes_put(S+h, K, T, r, sigma)
    price_down = black_scholes_put(S-h, K, T, r, sigma)
    delta_fd = (price_up - price_down)/(2*h)
    return delta_fd

def finite_difference_gamma(S, K, T, r, sigma, h = 0.01):
    call_price = black_scholes_call(S, K, T, r, sigma)
    call_price_up = black_scholes_call(S+h, K, T, r, sigma)
    call_price_down = black_scholes_call(S-h, K, T, r, sigma)
    gamma_fd = (call_price_up - (2*call_price) + call_price_down)/h**2
    return gamma_fd

def finite_difference_vega(S, K, T, r, sigma, h = 0.0001):
    price_up = black_scholes_call(S, K, T, r, sigma + h)
    price_down = black_scholes_call(S, K, T, r, sigma - h)
    vega_fd = (price_up - price_down)/(2*h)
    return vega_fd

def put_call_parity_check(S, K, T, r, sigma):
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    lhs = call_price - put_price
    rhs = S - K*np.exp(-r*T)
    return lhs, rhs 

def finite_difference_rho_call(S, K, T, r, sigma, h=0.01):
    vega_up = black_scholes_call(S, K, T, r+h, sigma)
    vega_down = black_scholes_call(S, K, T, r-h, sigma)
    return (vega_up - vega_down)/(2*h)

def finite_difference_rho_put(S, K, T, r, sigma, h=0.01):
    vega_up = black_scholes_put(S, K, T, r+h, sigma)
    vega_down = black_scholes_put(S, K, T, r-h, sigma)
    return (vega_up - vega_down)/(2*h)

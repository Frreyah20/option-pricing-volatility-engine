import numpy as np



def european_call_monte_carlo(S, K, T, r, sigma, n_simulations):
    Z = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T)*Z)
    payoffs = np.maximum(S_T - K, 0) #maximum because we are operating on an array
    discounted_payoffs = np.exp(-r*T) * payoffs
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof =1)/np.sqrt(n_simulations)
    lower_bound = option_price - 1.96*standard_error #for 95% ci
    upper_bound = option_price + 1.96*standard_error
    return option_price, lower_bound, upper_bound 

def european_put_monte_carlo(S, K, T, r, sigma, n_simulations):
    Z = np.random.standard_normal(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T)*Z)
    payoffs = np.maximum(K - S_T, 0) #maximum because we are operating on an array
    discounted_payoffs = np.exp(-r*T) * payoffs
    option_price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof =1)/np.sqrt(n_simulations)
    lower_bound = option_price - 1.96*standard_error #for 95% ci
    upper_bound = option_price + 1.96*standard_error
    return option_price, lower_bound, upper_bound 

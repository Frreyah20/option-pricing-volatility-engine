import numpy as np



def european_call_binomial_one_step(S, K, T, r, sigma):
    u = np.exp(sigma * np.sqrt(T)) #up move factor
    d = 1/u #down move factor
    p = (np.exp(r*T) - d)/(u-d) #risk neutral probability
    s_up = S*u
    s_down = S*d
    #payoffs
    call_up = max(s_up-K, 0)
    call_down = max(s_down-K, 0)
    expected_payoff = (p*call_up + (1-p)*call_down)
    
    call_price = np.exp(-r*T) * expected_payoff #discount back to today
    
    return call_price

def european_call_binomial(S, K, T, r, sigma, n_steps):
    dt = T/n_steps #length of each step
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    stock_prices = []
    for i in range(n_steps + 1):
        stock_price = (S * (u**(n_steps-i)) * (d**i))
        stock_prices.append(stock_price) 

    option_values = []
    for stock_price in stock_prices:
        payoff = max(stock_price - K, 0)
        option_values.append(payoff)

    for step in range(n_steps):
        new_option_values = []
        for i in range(len(option_values)-1):
            value = np.exp(-r*dt) * (p*option_values[i] + (1-p)*option_values[i+1])
            new_option_values.append(value)
        option_values = new_option_values

    return option_values[0]

def european_put_binomial(S, K, T, r, sigma, n_steps):
    dt = T/n_steps #length of each step
    u = np.exp(sigma * np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)
    stock_prices = []
    for i in range(n_steps + 1):
        stock_price = (S * (u**(n_steps-i)) * (d**i))
        stock_prices.append(stock_price) 

    option_values = []
    for stock_price in stock_prices:
        payoff = max(K-stock_price, 0)
        option_values.append(payoff)

    for step in range(n_steps):
        new_option_values = []
        for i in range(len(option_values)-1):
            value = np.exp(-r*dt) * (p*option_values[i] + (1-p)*option_values[i+1])
            new_option_values.append(value)
        option_values = new_option_values

    return option_values[0]
    


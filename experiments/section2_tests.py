from pricing.black_scholes import black_scholes_call
from pricing.binomial import european_call_binomial_one_step, european_call_binomial
from visualization.convergence_plot import plot_binomial_convergence, plot_monte_carlo_convergence
from pricing.monte_carlo import european_call_monte_carlo
import time


S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2
h = 0.01

bs_price = black_scholes_call(S, K, T, r, sigma)
bin_price = european_call_binomial_one_step(S, K, T,r, sigma)
print(f"Black Scholes call price = {bs_price:.4f}")
print(f"Binomial Call price = {bin_price:.4f}")

bin_price = european_call_binomial(S, K, T, r, sigma, 100)
print(f"Black Scholes call price = {bs_price:.4f}")
print(f"Binomial Call price = {bin_price:.4f}")

step_counts = [1, 2, 5, 10, 25, 50, 100, 250, 500]
errors = []
for n in step_counts:
    price = european_call_binomial(S, K, T, r , sigma, n)
    error = abs(price - bs_price)
    errors.append(error)
    print(f"Steps = {n:<4} "
          f"Price = {price:.4f} "
          f"Error = {error:.6f}")

plot_binomial_convergence(step_counts, errors)

bs_price = black_scholes_call(S, K, T, r, sigma)
print(f"Black Scholes call price = {bs_price:.4f}")
mc_price, lower_bound, upper_bound = european_call_monte_carlo(S, K, T, r, sigma, 10000)
print(f"Monte Carlo call price = {mc_price:.4f}")
print(f"95% confidence interval = ({lower_bound:.4f}, {upper_bound:.4f})")

simulation_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
errors = []
for n in simulation_counts:
    mc_price, lower_bound, upper_bound = european_call_monte_carlo(S, K, T, r, sigma, n)
    error = abs(mc_price - bs_price)
    errors.append(error)
    print(f"Simulation Count = {n:<7} "
        f"Price={mc_price:.4f} "
        f"Error={error:.6f}")

plot_monte_carlo_convergence(simulation_counts, errors)

start = time.perf_counter()
bs_price = black_scholes_call(S, K, T, r, sigma)
bs_runtime = time.perf_counter() - start 

start = time.perf_counter()
bin_price = european_call_binomial(S, K, T, r, sigma, 500)
bin_runtime = time.perf_counter() - start 

start = time.perf_counter()
mc_price, lower, upper = european_call_monte_carlo(S, K, T, r, sigma, 100000)
mc_runtime = time.perf_counter() - start

print("\nMethod Comparison")
print("-" * 60)

print(
    f"Black-Scholes | "
    f"Price={bs_price:.4f} | "
    f"Runtime={bs_runtime:.6f}s"
)

print(
    f"Binomial      | "
    f"Price={bin_price:.4f} | "
    f"Error={abs(bin_price-bs_price):.6f} | "
    f"Runtime={bin_runtime:.6f}s"
)

print(
    f"Monte Carlo   | "
    f"Price={mc_price:.4f} | "
    f"Error={abs(mc_price-bs_price):.6f} | "
    f"Runtime={mc_runtime:.6f}s"
)



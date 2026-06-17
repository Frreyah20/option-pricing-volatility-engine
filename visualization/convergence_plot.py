import matplotlib.pyplot as plt


def plot_binomial_convergence(step_counts, errors):
    plt.figure(figsize = (8, 5))
    plt.plot(step_counts, errors, marker = 'o')
    plt.xscale("log")
    plt.xlabel("Number of Steps")
    plt.ylabel("Absolute Error")
    plt.title("Binomial Convergence")
    plt.grid(True)
    plt.show()

def plot_monte_carlo_convergence(simulation_counts, errors):
    plt.figure(figsize = (8, 5))
    plt.plot(simulation_counts, errors, marker = 'o')
    plt.xscale("log")
    plt.xlabel("Number of Simulations")
    plt.ylabel("Absolute Error")    
    plt.title("Monte Carlo Convergence")
    plt.grid(True)
    plt.show()
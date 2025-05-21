from models.price_dynamics import generate_trajectories
from pricers.monte_carlo  import monte_carlo_pricer

def sensitivity_data(option, param, param_values, N, n_steps):
    prices = []
    original_value = getattr(option, param)

    for p in param_values:
        setattr(option, param, p) # dynamically set parameter like sigma, K, T etc.
        paths = generate_trajectories(N, n_steps, mu=option.r, sigma=option.sigma, S0=option.S0, T=option.T)
        mean_price, _ = monte_carlo_pricer(option, paths)
        prices.append(mean_price)

    setattr(option, param, original_value) # Restore original parameter 
    return param_values, prices
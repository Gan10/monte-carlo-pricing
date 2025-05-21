import numpy as np



def generate_trajectories(n_paths, n_steps, mu, sigma, S0, T, antithetic=False):
    Z = np.random.normal(size=(n_paths, n_steps))
    dt = T / n_steps

    def simulate(Z_):
        S = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_[:, t-1])
        return S

    if antithetic:
        return simulate(Z), simulate(-Z)
    else:
        return simulate(Z)

def generate_trajectories_importance_sampling(n_paths, n_steps, mu, sigma, S0, T, theta):
    
    Z = np.random.normal(size=(n_paths, n_steps))
    Z_shifted = Z + theta  
    dt = T / n_steps
    
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_shifted[:, t-1])
    
    # likelihood ratio
    sum_Z = np.sum(Z, axis=1)  #  sum of the original standard normals
    n = n_steps
    L = np.exp(-theta * sum_Z - 0.5 * n * theta**2)
    
    return S, L

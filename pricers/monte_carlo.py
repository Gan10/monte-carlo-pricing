import numpy as np

def monte_carlo_pricer(option, paths):

    payoffs = option.payoff(paths)
    

    discounted_payoff = np.exp(-option.r * option.T) * payoffs
    return np.mean(discounted_payoff), np.std(discounted_payoff) / np.sqrt(len(paths))


def monte_carlo_antithetic(option, paths1,paths2):

    payoffs1 = option.payoff(paths1)
    payoffs2 = option.payoff(paths2)
    payoffs = 0.5 * (payoffs1 + payoffs2)

    discounted_payoff = np.exp(-option.r * option.T) * payoffs
    return np.mean(discounted_payoff), np.std(discounted_payoff) / np.sqrt(len(paths1))

def monte_carlo_control_variate(exotic_option, control_option, paths):

    X = exotic_option.payoff(paths)
    Y = control_option.payoff(paths)
    

    c = control_option.black_scholes_price()
    
    cov_XY = np.cov(X, Y, ddof=1)[0,1]
    var_Y = np.var(Y, ddof=1)
    beta = cov_XY / var_Y
    
    Z = X - beta * (Y - c)
    
    discounted_Z = np.exp(-exotic_option.r * exotic_option.T) * Z
    
    price = np.mean(discounted_Z)
    std_error = np.std(discounted_Z) / np.sqrt(len(paths))
    
    return price, std_error, beta


def monte_carlo_importance_sampling(option, paths, weights):
    payoffs = option.payoff(paths)
    
    discounted_payoffs = np.exp(-option.r * option.T) * payoffs
    
    weighted_payoffs = discounted_payoffs * weights
    
    price = np.mean(weighted_payoffs)
    std_error = np.std(weighted_payoffs) / np.sqrt(len(paths))
    
    return price, std_error


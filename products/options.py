import numpy as np
from scipy.stats import norm

class Option:
    def __init__(self, S0, K, r, T, sigma, option_type="call"):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.sigma = sigma
        self.option_type = option_type.lower()
    
    def payoff(self, paths):
        raise NotImplementedError("Override payoff in subclass")


class EuropeanOption(Option):
    def payoff(self, paths):
        ST = paths[:, -1]
        if self.option_type == "call":
            return np.maximum(ST - self.K, 0)
        else:
            return np.maximum(self.K - ST, 0)

    def black_scholes_price(self):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.option_type == "call":
            price = self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        return price


class AsianOption(Option):
    def payoff(self, paths):
        avg_price = np.mean(paths[:, 1:], axis=1)  # exclude initial price
        if self.option_type == "call":
            return np.maximum(avg_price - self.K, 0)
        else:
            return np.maximum(self.K - avg_price, 0)
        
class LookbackOption(Option):
    def payoff(self, paths):
        ST = paths[:, -1]
        if self.option_type == "call":
            min_S = np.min(paths[:, 1:], axis=1)
            return np.maximum(ST - min_S, 0)
        else:
            max_S = np.max(paths[:, 1:], axis=1)
            return np.maximum(max_S - ST, 0)


class BarrierOption(Option):
    def __init__(self, S0, K, r, T, sigma, barrier, option_type="call", barrier_type="up-and-out"):
        super().__init__(S0, K, r, T, sigma, option_type)
        self.barrier = barrier
        self.barrier_type = barrier_type.lower()

    def payoff(self, paths):
        ST = paths[:, -1]
        max_S = np.max(paths[:, 1:], axis=1)
        min_S = np.min(paths[:, 1:], axis=1)
        
        if self.barrier_type == "up-and-out":
            alive = max_S < self.barrier
        elif self.barrier_type == "down-and-in":
            alive = min_S <= self.barrier
        else:
            raise ValueError("Unsupported barrier type")
        
        if self.option_type == "call":
            payoff = np.maximum(ST - self.K, 0)
        else:
            payoff = np.maximum(self.K - ST, 0)
        
        return payoff * alive

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

class PortfolioOptimizer:
    def __init__(self, returns):
        self.returns = returns
        self.mean_returns = returns.mean()
        self.cov_matrix = LedoitWolf().fit(returns).covariance_

    def optimize(self):
        num_assets = len(self.mean_returns)
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        return weights

    def calculate_portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        return portfolio_return, portfolio_std_dev
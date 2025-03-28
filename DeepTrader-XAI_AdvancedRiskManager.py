import numpy as np
import logging

class AdvancedRiskManager:
    def __init__(self, portfolio_value, confidence_level=0.95):
        self.portfolio_value = portfolio_value
        self.confidence_level = confidence_level

    def calculate_var(self, returns):
        mean = np.mean(returns)
        std_dev = np.std(returns)
        var = np.percentile(returns, (1 - self.confidence_level) * 100)
        return self.portfolio_value * (mean - var)

    def calculate_cvar(self, returns):
        var = self.calculate_var(returns)
        cvar = returns[returns <= var].mean()
        return self.portfolio_value * cvar
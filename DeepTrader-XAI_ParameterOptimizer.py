from sklearn.model_selection import RandomizedSearchCV

class ParameterOptimizer:
    def __init__(self, model, param_distributions, n_iter=10):
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def optimize(self, X, y):
        search = RandomizedSearchCV(self.model, self.param_distributions, n_iter=self.n_iter)
        search.fit(X, y)
        return search.best_params_
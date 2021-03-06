import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor


class MeanBaseEstimator:
    def __init__(self):
        self.y_mean = 0

    def fit(self, X, y):
        self.y_mean = np.mean(y)
        return self

    def predict(self, X):
        return np.ones(X.shape[0]) * self.y_mean


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, max_depth=3, max_features='sqrt',
                 min_samples_split=2, min_samples_leaf=1, learning_rate=1.0,
                 max_split_values=100):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_params = {
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_split_values': max_split_values
        }

    def fit(self, X, y):
        self.base = MeanBaseEstimator().fit(X, y)
        self.trees = []

        y_pred = self.base.predict(X)

        for i in range(self.n_estimators):
            error = y - y_pred

            t = DecisionTreeRegressor(**self.tree_params).fit(X, error)
            self.trees.append(t)

            y_pred += self.learning_rate * t.predict(X)

        return self

    def predict(self, X):
        y = self.base.predict(X)
        for t in self.trees:
            y += self.learning_rate * t.predict(X)
        return y


def prob_to_log_odds(p):
    return np.log(p / (1-p))


def log_odds_to_prob(lo):
    return 1 / (1 + np.exp(-lo))


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, max_depth=3, max_features='sqrt',
                 min_samples_split=2, min_samples_leaf=1, learning_rate=1.0,
                 max_split_values=100):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.tree_params = {
            'max_depth': max_depth,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_split_values': max_split_values
        }

    def fit(self, X, y):
        self.base = MeanBaseEstimator().fit(X, y)
        self.trees = []

        y_pred = self.base.predict(X)
        lo = prob_to_log_odds(y_pred)

        for i in range(self.n_estimators):
            error = y - y_pred  # gradient of cross entropy wrt log odds

            t = DecisionTreeRegressor(**self.tree_params).fit(X, error)
            self.trees.append(t)

            lo += self.learning_rate * t.predict(X)
            y_pred = log_odds_to_prob(lo)

        return self

    def predict(self, X):
        y = self.base.predict(X)
        lo = prob_to_log_odds(y)
        for t in self.trees:
            lo += self.learning_rate * t.predict(X)
        return log_odds_to_prob(lo)

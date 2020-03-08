import random
import numpy as np


class Node:
    def __init__(self, col=-1, value=None,
                 pos_branch=None, neg_branch=None):
        self.col = col
        self.value = value
        self.pos_branch = pos_branch
        self.neg_branch = neg_branch


class Leaf:
    def __init__(self, value):
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, max_depth=None, max_features=None,
                 min_samples_split=2, min_samples_leaf=1,
                 max_split_values=100):
        self.root_node = None
        self.max_depth = max_depth
        self.feature_indices = []
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_split_values = max_split_values

    def fit(self, X, y):
        if self.max_features is not None:
            self.feature_indices = self.sample_features(X)
            X = X[:, self.feature_indices]

        if self.max_depth is None:
            self.max_depth = -1

        self.root_node = self.build_tree(X, y, self.max_depth)

        return self

    def predict(self, X):
        if self.max_features is not None:
            X = X[:, self.feature_indices]

        y = np.array([self.predict_sample(x, self.root_node) for x in X])

        return y

    def sample_features(self, X):
        if self.max_features == 'sqrt':
            return random.sample(range(X.shape[1]), int(np.sqrt(X.shape[1])))
        else:
            raise ValueError('Unknown value for max_features')

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def build_tree(self, X, y, depth):
        if len(X) == 0:
            return Node()
        if depth == 0 or len(X) < self.min_samples_split:
            return Leaf(value=np.mean(y))

        current_mse = self.mse(y)
        best_reduction = 0

        for col in range(X.shape[1]):
            values = np.unique(X[:, col])
            if len(values) > self.max_split_values:
                percentiles = np.linspace(0, 100, num=self.max_split_values)
                values = np.percentile(values, percentiles)
            for value in values:
                pos_rows = X[:, col] >= value
                neg_rows = ~pos_rows

                n_pos = np.sum(pos_rows)
                n_neg = np.sum(neg_rows)

                p = n_pos / len(X)
                new_mse = (p * self.mse(y[pos_rows]) +
                           (1-p) * self.mse(y[neg_rows]))
                mse_reduction = current_mse - new_mse

                if (mse_reduction > best_reduction and
                    n_pos >= self.min_samples_leaf and
                    n_neg >= self.min_samples_leaf):
                    best_reduction = mse_reduction
                    best_col = col
                    best_value = value
                    best_pos_rows = pos_rows
                    best_neg_rows = neg_rows

        if best_reduction > 0:
            pos_branch = self.build_tree(X[best_pos_rows], y[best_pos_rows], depth-1)
            neg_branch = self.build_tree(X[best_neg_rows], y[best_neg_rows], depth-1)
            return Node(col=best_col, value=best_value,
                        pos_branch=pos_branch, neg_branch=neg_branch)
        else:
            return Leaf(value=np.mean(y))

    def predict_sample(self, x, node):
        if isinstance(node, Leaf):
            return node.value
        if x[node.col] >= node.value:
            return self.predict_sample(x, node.pos_branch)
        else:
            return self.predict_sample(x, node.neg_branch)

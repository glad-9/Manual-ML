import numpy as np
from data_processing.base import Transformer

class Encoder(Transformer):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        self.categories = {}
        for col in self.columns:
            self.categories[col] = np.unique(X[:, col])

    def transform(self, X):
        X = X.copy()
        encoded_parts = []
        for col in self.columns:
            for category in self.categories[col]:
                encoded_parts.append((X[:, col] == category).astype(float).reshape(-1,1))

        X = np.delete(X, self.columns, axis=1)
        return np.h_stack([X] + encoded_parts)

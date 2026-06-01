import numpy as np
from data_processing.base import Transformer

class Normalizer(Transformer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = np.sqrt(X.var(axis=0) + self.epsilon)

    def transform(self, X):
        return (X - self.mean) / self.std

    

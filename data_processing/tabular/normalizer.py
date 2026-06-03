import numpy as np
from data_processing.base import Transformer

class Normalizer(Transformer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.means = None
        self.stds = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = np.sqrt(X.var(axis=0) + self.epsilon)

    def __call__(self, x):
        return (x - self.mean) / self.std

    

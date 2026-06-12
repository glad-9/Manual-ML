import numpy as np
from data_processing.base import Transformer


class ImageNormalizer(Transformer):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
        self.std = X.std(axis=(0, 2, 3), keepdims=True) + self.epsilon

    def __call__(self, x):
        return (x - self.mean[0]) / self.std[0]

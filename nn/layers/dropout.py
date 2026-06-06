import numpy as np
from .base import Layer

class Dropout(Layer):
    def __init__(self, dropout_rate):
        self.rate = dropout_rate

    def forward(self, X):
        if self.is_training():
            self.mask = np.random.randn(*X.shape) > self.rate
            return X * self.mask / (1 - self.rate)
        return X

    def get_params(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass
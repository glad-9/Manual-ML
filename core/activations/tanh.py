import numpy as np
from core.layers.base import Layer

class Tanh(Layer):

    def forward(self, X):
        self.out = (np.exp(X) - np.exp(-X))/ (np.exp(X) + np.exp(-X))
        return self.out

    def backward(self, grad):
        return grad * (1 - (self.out ** 2))

    def get_params_and_grads(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass
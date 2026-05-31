import numpy as np
from core.layers.base import Layer

class ReLU(Layer):
    def forward(self, X):
        self.out = np.maximum(0, X)
        return self.out

    def backward(self, grad):
        return grad * (self.out > 0)

    def get_params_and_grads(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass
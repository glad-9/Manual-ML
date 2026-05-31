import numpy as np
from core.layers.base import Layer

class Sigmoid(Layer):

    def forward(self, X):
        self.out = 1 / (1 + (np.exp(-X)))
        return self.out

    def backward(self, grad):
        return grad * (self.out * (1 - self.out))

    def get_params_and_grads(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass
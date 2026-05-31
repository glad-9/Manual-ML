import numpy as np
from .base import Activation

class Tanh(Activation):

    def forward(self, X):
        self.out = (np.exp(X) - np.exp(-X))/ (np.exp(X) + np.exp(-X))
        return self.out

    def backward(self, grad):
        return grad * (1 - (self.out ** 2))

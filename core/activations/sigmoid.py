import numpy as np
from .base import Activation

class Sigmoid(Activation):

    def forward(self, X):
        self.out = 1 / (1 + (np.exp(-X)))
        return self.out

    def backward(self, grad):
        return grad * (self.out * (1 - self.out))

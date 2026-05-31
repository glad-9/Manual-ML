import numpy as np
from .base import Activation

class ReLU(Activation):
    def forward(self, X):
        self.out = np.maximum(0, X)
        return self.out

    def backward(self, grad):
        return grad * (self.out > 0)

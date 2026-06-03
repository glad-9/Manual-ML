import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, layers):
        for layer in layers:
            for p, regularize in layer.get_params():
                if regularize:
                    p.grad += self.reg * p.data

                p.data -= self.lr * p.grad
                p.grad = np.zeros_like(p.data)

                




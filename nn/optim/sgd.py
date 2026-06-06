import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, layers):
        for layer in layers:
            for p in layer.get_params():
                if p.requires_reg:
                    p.grad += self.reg * p.data

                p.data -= self.lr * p.grad
                p.zero_grad()

                




import numpy as np
from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, layers):
        for layer in layers:
            for param, grad, regularize in layer.get_params_and_grads():
                if regularize:
                    grad += self.reg * param
                
                grad = np.clip(grad, -5, 5)

                param[:] -= self.lr * grad


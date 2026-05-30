from .base import Optimizer
import numpy as np

class Adagrad(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr
        self.sq_grads = {} # Maintains a running average of squared gradients
        self.epsilon = 1e-8

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.get_params_and_grads():
                key = id(param)
                grad = np.clip(grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = np.zeros_like(param)

                self.sq_grads[key] += grad ** 2
                param[:] -= ((self.lr * grad)/np.sqrt(self.sq_grads[key] + self.epsilon))
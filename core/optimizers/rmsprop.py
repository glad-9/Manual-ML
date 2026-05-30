from .base import Optimizer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, lr=0.01, dr=0.9):
        self.lr = lr
        self.dr = dr # Decay Rate
        self.sq_grads = {} # Maintains a running average of squared gradients
        self.epsilon = 1e-8

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.get_params_and_grads():
                key = id(param)
                grad = np.clip(grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = np.zeros_like(param)

                v_prev = self.sq_grads[key]

                v_next = (self.dr * v_prev) + (1 - self.dr) * (grad ** 2)

                self.sq_grads[key] = v_next
                param[:] -= ((self.lr * grad)/np.sqrt(v_next + self.epsilon))
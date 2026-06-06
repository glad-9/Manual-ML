from .base import Optimizer
import numpy as np

class Adagrad(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.sq_grads = {} # Maintains a running average of squared gradients
        self.epsilon = 1e-8

    def step(self, layers):
        for layer in layers:
            for p in layers.get_params():
                key = id(p)
                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = np.clip(p.grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = np.zeros_like(p.data)

                self.sq_grads[key] += grad ** 2
                p.data -= ((self.lr * grad)/np.sqrt(self.sq_grads[key] + self.epsilon))
                p.zero_grad()
from .base import Optimizer
import numpy as np

class RMSprop(Optimizer):
    def __init__(self, dr=0.99, **kwargs):
        super().__init__(**kwargs)

        self.dr = dr # Decay Rate
        self.sq_grads = {} # Maintains a running average of squared gradients
        self.epsilon = 1e-8

    def step(self, layers):
        for layer in layers:
            for p in layer.get_params():
                key = id(p)
                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = np.clip(p.grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = np.zeros_like(p.data)

                v_prev = self.sq_grads[key]

                v_next = (self.dr * v_prev) + (1 - self.dr) * (grad ** 2)

                self.sq_grads[key] = v_next
                p.data -= ((self.lr * grad)/np.sqrt(v_next + self.epsilon))
                p.zero_grad()
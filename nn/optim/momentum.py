from .base import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, mf=0.9, **kwargs):
        super().__init__(**kwargs)

        self.mf = mf # mf = momentum factor
        self.momentum = {} # Maintains a running average of all gradients

    def step(self, layers):
        for layer in layers:
            for p in layers.get_params():
                key = id(p)

                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = np.clip(p.grad, -5, 5)

                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(p.data)

                m_prev = self.momentum[key]

                m_next = (self.mf * m_prev) + (1 - self.mf) * grad

                self.momentum[key] = m_next
                p.data -= self.lr * m_next
                p.grad = None


                
                

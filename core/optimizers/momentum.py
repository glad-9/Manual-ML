from .base import Optimizer
import numpy as np

class Momentum(Optimizer):
    def __init__(self, lr=0.01, mf=0.9):
        self.lr = lr
        self.mf = mf # mf = momentum factor
        self.momentum = {} # Maintains a running average of all gradients

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.get_params_and_grads():
                key = id(param)
                grad = np.clip(grad, -5, 5)

                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(param)

                m_prev = self.momentum[key]

                m_next = (self.mf * m_prev) + (1 - self.mf) * grad

                self.momentum[key] = m_next
                param[:] -= self.lr * m_next


                
                

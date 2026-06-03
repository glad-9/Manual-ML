from .base import Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, mf=0.9, dr=0.99, **kwargs):
        super().__init__(**kwargs)

        self.mf = mf # mf = momentum factor
        self.dr = dr # dr = decay rate
        self.momentum = {} # Maintains a running average of gradients
        self.velocity = {} # Maintains a running average of squared gradients
        self.t = 0 # Step counter
        self.epsilon = 1e-8

    def step(self, layers):
        self.t += 1
        for layer in layers:
            for param, grad, regularize in layer.get_params_and_grads():

                if regularize:
                    grad += self.reg * param
                
                key = id(param)
                grad = np.clip(grad, -5, 5)

                # First moment estimate

                if key not in self.momentum:
                    self.momentum[key] = np.zeros_like(param)

                m_prev = self.momentum[key]

                m_raw = (self.mf * m_prev) + (1 - self.mf) * grad

                self.momentum[key] = m_raw

                # Second moment estimate
                if key not in self.velocity:
                    self.velocity[key] = np.zeros_like(param)

                v_prev = self.velocity[key]

                v_raw = (self.dr * v_prev) + (1 - self.dr) * (grad ** 2)

                self.velocity[key] = v_raw

                # Bias correction
                m_corrected = m_raw/(1 - self.mf ** self.t)
                v_corrected = v_raw/(1 - self.dr ** self.t)

                param[:] -= ((self.lr * m_corrected)/(np.sqrt(v_corrected) + self.epsilon))
            
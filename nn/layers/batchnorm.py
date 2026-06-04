import numpy as np
from core.tensor import Tensor
from .base import Layer

class BatchNorm(Layer):
    def __init__(self, features, momentum=0.9, epsilon=1e-8):
        self.gamma = Tensor(np.ones((1, features)).astype('float32'), requires_grad=True) # Learnable scale
        self.beta =  Tensor(np.zeros((1, features)).astype('float32'), requires_grad=True) # Learnable shift
        self.momentum = momentum
        self.epsilon = epsilon

        # Used in inference
        self.running_mean = Tensor(np.zeros((1, features)).astype('float32'))
        self.running_var = Tensor(np.zeros((1, features)).astype('float32'))

    def forward(self, X):
        if self.is_training():
            # mean = (1/m) * sum(X)
            self.mean = X.mean()

            # var = (1/m) * sum((X - mean) ^ 2)
            self.var = X.var()

            self.X_norm = (X - self.mean) / (self.var + self.epsilon) ** 0.5

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            return self.gamma * self.X_norm + self.beta 

        else:
            X_norm = (X - self.running_mean) / ((self.running_var + self.epsilon)) ** 0.5
            return self.gamma * X_norm + self.beta

    def get_params(self):
        return [self.gamma, self.beta]

    def save_state(self):
        return {
            "gamma": self.gamma.data.copy(),
            "beta": self.beta.data.copy(),
            "running_mean": self.running_mean.data.copy(),
            "running_var": self.running_var.data.copy()
        }

    def load_state(self, state):
        self.gamma = Tensor(state["gamma"].copy(), requires_grad=True)
        self.beta = Tensor(state["beta"].copy(), requires_grad=True)
        self.running_mean = Tensor(state["running_mean"].copy())
        self.running_var = Tensor(state["running_var"].copy())
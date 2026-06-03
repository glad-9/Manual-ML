import numpy as np
from .base import Layer

class BatchNorm(Layer):
    def __init__(self, features, momentum=0.9, epsilon=1e-8):
        self.gamma = np.ones((1, features)) # Learnable scale
        self.beta =  np.zeros((1, features)) # Learnable shift
        self.momentum = momentum
        self.epsilon = epsilon

        # Used in inference
        self.running_mean = np.zeros((1, features))
        self.running_var = np.zeros((1, features))

    def forward(self, X):
        self.X = X
        if self.is_training():

            # mean = (1/m) * sum(X)
            self.mean = X.mean(axis=0)

            # var = (1/m) * sum((X - mean) ^ 2)
            self.var = X.var(axis=0)

            self.X_norm = (X - self.mean) / np.sqrt(self.var + self.epsilon)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            return self.gamma * self.X_norm + self.beta 

        else:
            X_norm = (X - self.running_mean) / np.sqrt((self.running_var + self.epsilon))
            return self.gamma * X_norm + self.beta


    def backward(self, grad):

        # Passback: dL/dX = dL/dX_norm * dX_norm/dX + dL/dvar * dvar/dX + dL/dmean + dmean/dX

        # grad -> dL/dout
        m = grad.shape[0] # Features

        # dL/dX_norm = dL/dout * dout/dX_norm
        dX_norm = grad * self.gamma

        # dL/dvar = dL/dX_norm * dX_norm/dvar
        dvar = np.sum(dX_norm * (self.X - self.mean) * (-0.5) * ((self.var + self.epsilon) ** (-1.5)), axis = 0) 

        # dL/dmean = dL/dX_norm * dX_norm/dmean + dL/dvar * dvar/dmean
        dmean = (1/m) * np.sum(dX_norm * ((-1) / np.sqrt(self.var + self.epsilon)), axis=0) \
             + dvar * np.sum(-2 * (self.X - self.mean), axis=0) / m

        # dL/dX: three paths - direct, through var, through mean
        dX = dX_norm * (1 / np.sqrt(self.var + self.epsilon)) \
            + dvar * ((1/m) * 2 * (self.X - self.mean)) \
            + dmean * (1/m)
            
        """
        dL/dgamma = dL/dout * dout/dgamma = grad * self.X_norm
        dL/dbeta = dL/dout * dout/dbeta = grad * (1)
        """
        self.dgamma = np.sum(grad * self.X_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(grad, axis=0, keepdims=True)

        return dX

    def get_params_and_grads(self):
        return [(self.gamma, self.dgamma, False), (self.beta, self.dbeta, False)]

    def save_state(self):
        return {
            "gamma": self.gamma.copy(),
            "beta": self.beta.copy(),
            "running_mean": self.running_mean.copy(),
            "running_var": self.running_var.copy()
        }

    def load_state(self, state):
        self.gamma = state["gamma"].copy()
        self.beta = state["beta"].copy()
        self.running_mean = state["running_mean"].copy()
        self.running_var = state["running_var"].copy()
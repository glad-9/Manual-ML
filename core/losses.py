import numpy as np

def mse(y_hat, y):
    cost = np.mean(0.5 * ((y_hat - y) ** 2))
    return cost

def mse_deriv(y_hat, y):
    cost_deriv = (y_hat-y)
    return cost_deriv

def bce(y_hat, y):
    y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
    loss = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
    return loss

def bce_deriv(y_hat, y):
    y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
    loss_deriv = ((y_hat - y) / (y_hat * (1 - y_hat)))
    return loss_deriv


class Loss:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward
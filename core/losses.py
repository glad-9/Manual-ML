import numpy as np

class BCE:
    def __init__(self):
        self.name = "bce"

    def forward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
        self.loss = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
        return self.loss

    def backward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
        self.deriv = ((y_hat - y) / (y_hat * (1 - y_hat)))
        return self.deriv


class MSE:
    def __init__(self):
        self.name = "mse"

    def forward(self, y_hat, y):
        self.cost = np.mean(0.5 * ((y_hat - y) ** 2))
        return self.cost

    def backward(self, y_hat, y):
        self.deriv = (y_hat) - y
        return self.deriv


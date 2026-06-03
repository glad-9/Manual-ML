import numpy as np

class MSE:
    name = "mse"

    def forward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        self.cost = np.mean(0.5 * ((y_hat - y) ** 2))
        return self.cost

    def backward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

        batch_size = y.shape[0]
        self.deriv = ((y_hat) - y) / batch_size
        return self.deriv
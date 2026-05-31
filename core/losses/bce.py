import numpy as np

class BCE:
    name = "bce"

    def forward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1-1e-15)
        self.loss = -np.mean(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat))
        return self.loss

    def backward(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-15, 1-1e-15)

        batch_size = y.shape[0]
        self.deriv = ((y_hat - y) / (y_hat * (1 - y_hat))) / batch_size
        return self.deriv


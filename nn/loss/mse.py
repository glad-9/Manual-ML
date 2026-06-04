import numpy as np

class MSE:
    name = "mse"

    def forward(self, y_hat, y):
        return y_hat.mse(y)

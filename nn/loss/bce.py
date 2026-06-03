import numpy as np

class BCE:
    name = "bce"

    def forward(self, y_hat, y):
        return y_hat.bce(y)


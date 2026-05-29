import numpy as np

class Linear:
    name = "linear"

    def forward(self, z):
        return z

    def backward(self, z):
        return 1

class ReLU:
    name = "relu"

    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, z):
        return (z > 0).astype(float)

class Sigmoid:
    name = "sigmoid"

    def forward(self, z):
        return 1 / (1 + (np.exp(-z)))

    def backward(self, z):
        s = self.forward(z)
        return s * (1 - s)




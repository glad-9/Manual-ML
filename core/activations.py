import numpy as np

def linear_activation(z):
    return z


def reLU(z):
    return np.maximum(0, z)


def sigmoid(z):
    return 1 / (1 + (np.exp(-z)))


def linear_deriv(z):
    return 1


def reLU_deriv(z):
    return (z > 0).astype(float)


def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

class Activation:
    def __init__(self, forward, backward):
        self.forward = forward
        self.backward = backward


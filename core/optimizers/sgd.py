from .base import Optimizer

class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            for param, grad in layer.get_params_and_grads():
                param[:] -= self.lr * grad


import numpy as np

from core.tensor import Tensor
from .base import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size, initializer):
        self.input_size = input_size
        self.output_size = output_size

        self.shape = (input_size, output_size)
        
        self.W = Tensor(initializer.initialize(self.shape), requires_grad=True)
        self.B = Tensor(np.zeros((1, output_size)), requires_grad=True)

    def forward(self, X):
        X = X if isinstance(X, Tensor) else Tensor(X)
        return X @ self.W + self.B

    def get_params(self):
        return [(self.W, True), (self.B, False)]

    def save_state(self):
        return {
            "W" : self.W.data.copy(),
            "B" : self.B.data.copy()
        }

    def load_state(self, state):
        self.W  = Tensor(state["W"].copy(), requires_grad=True)
        self.B  = Tensor(state["B"].copy(), requires_grad=True)




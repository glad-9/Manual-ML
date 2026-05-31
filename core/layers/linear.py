import numpy as np
from .base import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size, initializer):
        self.input_size = input_size
        self.output_size = output_size

        self.shape = (input_size, output_size)
        
        self.W = initializer.initialize(self.shape)
        self.B = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.B
        return self.Z

    def backward(self, grad):
        # self.dZ = grad * (activation derivative = 1 because the layer is linear) = grad
        # self.dW = self.X.T @ self.dZ -> self.X is the previous activation, self.dZ = grad
        self.dW = self.X.T @ grad

        # self.dB = self.dZ = grad
        self.dB = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T # grad passback

    def get_params_and_grads(self):
        return [(self.W, self.dW,True), (self.B,self.dB,False)]

    def save_state(self):
        return {
            "W" : self.W.copy(),
            "B" : self.B.copy()
        }

    def load_state(self, state):
        self.W  = state["W"].copy()
        self.B  = state["B"].copy()




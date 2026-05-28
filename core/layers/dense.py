import numpy as np

class Dense:
    def __init__(self, input_size, count, activation):
        self.input_size = input_size
        self.count = count
        self.a_func = activation.forward
        self.a_deriv = activation.backward

        # self.W = np.random.randn(input_size, count)
        # He / Xavier Scaling
        self.W = np.random.randn(input_size, count) * np.sqrt(2.0 / input_size)
        self.B = np.zeros((1, count))

    def activate(self, a_prev):
        self.a_prev = a_prev # shape: (samples, input_size)
        self.Z = self.a_prev @ self.W + self.B # (samples, count)
        self.A = self.a_func(self.Z)
        return self.A

    def compute_gradients(self, dA_back):
        # Number of examples in the current batch
        m = self.a_prev.shape[0]

        # Element-wise multiplication
        self.dZ = dA_back * self.a_deriv(self.Z)  # (samples, count)
        # print(dA_back.shape)
        # print(self.a_deriv(self.Z).shape)
        # print(self.dZ.shape)

        #
        self.dW = self.a_prev.T @ self.dZ / m   # (count_prev, samples) @ (samples, count)  = (count_prev, count) AKA (input_size, count) = Shape(W)
        self.dB = np.sum(self.dZ, axis=0, keepdims=True) / m

        #
        self.dA_backpass = self.dZ @ self.W.T  # (samples, count) @ (count, input_size) = (samples, input_size) AKA (samples, count_prev) = Shape(a_prev)

        return self.dA_backpass

    def update_params(self, lr):
        self.W -= lr * self.dW
        self.B -= lr * self.dB





    

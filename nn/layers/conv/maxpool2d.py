from layers.base import Layer


class MaxPool2D(Layer):
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        return X.pool2d(self.pool_size, self.stride)

    def get_params(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass

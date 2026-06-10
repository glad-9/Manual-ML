from nn.layers.base import Layer


class Flatten(Layer):
    def forward(self, X):
        n = X.data.shape[0]
        return X.reshape(n, -1)

    def get_params(self):
        return []

    def save_state(self):
        return {}

    def load_state(self, state):
        pass

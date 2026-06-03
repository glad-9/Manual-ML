from .base import Activation

class Tanh(Activation):
    def forward(self, X):
        return X.tanh()


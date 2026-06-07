from .base import Activation

class ReLU(Activation):
    def forward(self, X):
        return X.relu()

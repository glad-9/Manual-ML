from .base import Activation

class Sigmoid(Activation):
    def forward(self, X):
        return X.sigmoid()

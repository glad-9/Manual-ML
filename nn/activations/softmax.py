from .base import Activation

class Softmax(Activation):
    def forward(self, X):
        return X.softmax()
        
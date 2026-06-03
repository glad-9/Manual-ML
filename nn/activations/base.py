from nn.layers.base import Layer
from abc import ABC, abstractmethod

class Activation(Layer):

    def save_state(self):
        return {}

    def load_state(self, state):
        pass
from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    @abstractmethod
    def get_params_and_grads(self):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self, state):
        pass

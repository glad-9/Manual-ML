from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad):
        pass

    @abstractmethod
    def update_params(self, optimizer):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

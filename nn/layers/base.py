from abc import ABC, abstractmethod
from core.tensor import Tensor


class Layer(ABC):
    training: bool = True

    @classmethod
    def set_training(cls, mode: bool):
        cls.training = mode

    @classmethod
    def is_training(cls) -> bool:
        return Layer.training

    @abstractmethod
    def forward(self, X):
        return Tensor(...)

    @abstractmethod
    def save_state(self):
        return {}

    @abstractmethod
    def load_state(self, state):
        pass

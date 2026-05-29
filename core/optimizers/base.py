from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update(self, layer):
        pass
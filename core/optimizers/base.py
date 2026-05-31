from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, lr: float = 0.01, reg: float = 0.0):
        self.lr = lr
        self.reg = reg

    @abstractmethod
    def step(self, layers):
        pass
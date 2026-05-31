import numpy as np
from abc import ABC, abstractmethod

class Initializer(ABC):

    def __init__(self, seed=69):
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def initialize(self, shape):
        pass
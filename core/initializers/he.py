import numpy as np
from .base import Initializer

class He(Initializer):
    def initialize(self, shape):
        std = np.sqrt(2.0 / shape[0])
        return self.rng.standard_normal(size=shape) * std

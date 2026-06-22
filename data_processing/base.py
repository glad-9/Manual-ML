import numpy as np
from core.tensor import Tensor
from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def apply_transforms(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class Subset(Dataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def get_all(self):
        X = np.stack([self.dataset[i][0] for i in self.indices])
        y = np.stack([self.dataset[i][1] for i in self.indices])
        return X, y


class Transformer(ABC):
    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def __call__(self, x):
        pass


class Pipeline(ABC):
    @abstractmethod
    def run(self, **kwargs):
        pass

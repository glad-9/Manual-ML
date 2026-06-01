import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

class Subset(Dataset):
    def __init__(self, dataset, indices):
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
    def transform(self, X):
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class Pipeline(ABC):
    @abstractmethod
    def run(self, **kwargs):
        pass
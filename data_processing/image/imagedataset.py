import numpy as np
from data_processing.base import Dataset


class ImageDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        super().__init__()
        self.X = X
        self.y = y
        self.transforms = transforms or []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        x = self.apply_transforms(x)
        return x, self.y[idx]

    def get_all(self):
        samples = [self[i] for i in range(len(self))]
        X = np.stack([s[0] for s in samples])
        y = np.stack([s[1] for s in samples])
        return X, y

    @property
    def n_features(self):
        return self.X.shape[1:]

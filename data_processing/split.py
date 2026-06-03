import numpy as np
from .base import Dataset, Subset


def split(dataset, train_ratio=0.6, val_ratio=0.2, seed=69):
    """
    Splits dataframe into train/validation/test subsets.
    """

    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(dataset))

    train_end = int(len(dataset) * train_ratio)
    val_end = train_end + int(len(dataset) * val_ratio)

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return (train_set, val_set, test_set)
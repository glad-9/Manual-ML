import pandas as pd
import numpy as np


def train_test_split(df, train_ratio=0.6, val_ratio=0.2, seed=69):
    """
    Splits dataframe into train/validation/test subsets.
    """

    np.random.seed(seed)

    shuffled_indices = np.random.permutation(len(df))

    train_end = int(len(df) * train_ratio)
    val_end = train_end + int(len(df) * val_ratio)

    train_indices = shuffled_indices[:train_end]
    val_indices = shuffled_indices[train_end:val_end]
    test_indices = shuffled_indices[val_end:]

    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)

    return train_df, val_df, test_df
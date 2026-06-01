import numpy as np
from data_processing.base import Dataset

class TabularDataset(Dataset):
    def __init__(self, df, target_column, drop_columns=None):
        self.df = df
        if drop_columns:
            self.df = self.df.drop(columns=drop_columns)
                
        self.X = self.df.drop(columns=[target_column]).to_numpy()
        self.y = df[target_column].to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_all(self):
        return self.X, self.y
import numpy as np
from data_processing.base import Pipeline
from data_processing.split import split
from data_processing.loaders import load_csv
from data_processing.tabular.tabulardataset import TabularDataset
from data_processing.tabular.normalizer import Normalizer
from data_processing.tabular.encoder import Encoder



class TabularPipeline(Pipeline):
    def __init__(self, path, transformers, train_ratio, val_ratio):
        self.path = path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.transformers = transformers or []

    def run(self, target_column, drop_columns=None):
        df = load_csv(self.path)

        dataset = TabularDataset(df, target_column, drop_columns)

        train_set, val_set, test_set = split(dataset, self.train_ratio, self.val_ratio)

        X_train, y_train = train_set.get_all()
        X_val, y_val = val_set.get_all()
        X_test, y_test = test_set.get_all()

        for transformer in self.transformers:
            X_train = transformer.fit_transform(X_train)
            X_val = transformer.transform(X_val)
            X_test = transformer.transform(X_test)

        y_train = y_train.reshape(-1,1)
        y_val = y_val.reshape(-1,1)
        y_test = y_test.reshape(-1,1)

        return {
            "train": (X_train, y_train),
            "cv": (X_val, y_val),
            "test": (X_test, y_test)
        }





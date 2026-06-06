import numpy as np
from data_processing.base import Pipeline
from data_processing.split import split
from data_processing.loaders import load_csv
from data_processing.tabular.tabulardataset import TabularDataset
from data_processing.tabular.normalizer import Normalizer
from data_processing.tabular.encoder import Encoder



class TabularPipeline(Pipeline):
    def __init__(
        self,
        path,
        target_column,
        categorical_columns=None,
        drop_columns=None,
        train_ratio=0.8,
        val_ratio=0.1,
        normalize=True,
    ):
        self.path = path
        self.target_column = target_column
        self.categorical_columns = categorical_columns or []
        self.drop_columns = drop_columns or []
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize
 
    def run(self):
        # 1. load
        df = load_csv(self.path)
 
        if self.drop_columns:
            df = df.drop(columns=self.drop_columns)
 
        # 2. encode categorical columns at dataframe stage
        if self.categorical_columns:
            encoder = Encoder(self.categorical_columns)
            encoder.fit(df.drop(columns=[self.target_column]))
            df = encoder.transform_df(df)

        label_vals = sorted(df[self.target_column].unique())
        if not set(label_vals).issubset({0, 1, 0.0, 1.0}):
            df[self.target_column] = df[self.target_column].map(
                {label_vals[0]: 0.0, label_vals[1]: 1.0}
            )

        # 3. convert to numpy and build dataset
        target = df[self.target_column].to_numpy(dtype=np.float32)
        features = df.drop(columns=[self.target_column]).to_numpy(dtype=np.float32)
 
        # use a temporary dataset just for splitting
        # transforms not attached yet — normalizer not fitted
        raw_dataset = TabularDataset(features, target)
        train_set, val_set, test_set = split(raw_dataset, self.train_ratio, self.val_ratio)
 
        # 4. fit normalizer on train features only
        if self.normalize:
            X_train, _ = train_set.get_all()  # no transforms yet, raw values
            normalizer = Normalizer()
            normalizer.fit(X_train)
 
            # 5. attach as per-sample transform on all splits
            # val and test share the same underlying dataset so we need
            # to apply transforms at the Subset level instead
            X_train_raw, y_train = train_set.get_all()  # still raw, no transforms yet
            X_val_raw, y_val = val_set.get_all()
            X_test_raw, y_test = test_set.get_all()

            train_set = TabularDataset(X_train_raw, y_train, transforms=[normalizer])
            val_set = TabularDataset(X_val_raw, y_val, transforms=[normalizer])
            test_set = TabularDataset(X_test_raw, y_test, transforms=[normalizer])
 
        return {
            "train": train_set,
            "cv": val_set,
            "test": test_set,
            "n_features": raw_dataset.n_features,
        }
 
 






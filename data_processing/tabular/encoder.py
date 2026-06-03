import numpy as np
import pandas as pd
from data_processing.base import Transformer

class Encoder(Transformer):
    def __init__(self, columns):
        self.columns = columns 
        self.categories = {}
        self.col_indices = {} 
 
    def fit(self, df):
        # fit works on the dataframe before numpy conversion
        for col in self.columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")
            self.categories[col] = sorted(df[col].unique())
 
    def transform_df(self, df):
        df = df.copy()
        encoded_parts = []
 
        for col in self.columns:
            for category in self.categories[col]:
                encoded_col = f"{col}_{category}"
                encoded_parts.append(
                    pd.Series((df[col] == category).astype(float), name=encoded_col)
                )
 
        df = df.drop(columns=self.columns)
        return pd.concat([df] + [p.to_frame() for p in encoded_parts], axis=1)
 
    def __call__(self, x):
        # encoding is done at the dataframe stage, not per sample
        # this is a no-op at sample level
        return x


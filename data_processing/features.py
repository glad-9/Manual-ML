import numpy as np

def split_features_labels(df, target_column):
    """
    Separate features and labels.
    """
    y = df[target_column].to_numpy()

    X = df.drop(columns=[target_column]).to_numpy()

    return X, y

def reshape_inputs(X, y):
    """
    Converts shapes into neural-network format.
    X: (samples, features)
    Y: (samples,) -> (samples, 1)
    """

    y = y.reshape(-1,1)

    return X, y
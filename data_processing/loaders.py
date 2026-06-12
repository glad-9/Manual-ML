import numpy as np
import pandas as pd
import idx2numpy


def load_csv(file):
    df = pd.read_csv(file)
    return df


def load_idx(image_file, label_file):
    X = idx2numpy.convert_from_file(image_file)  # (N, H, W)
    y = idx2numpy.convert_from_file(label_file)  # (N,)

    X = X[:, np.newaxis, :, :].astype(np.float32)  # (N, 1, H, W)
    y = y.astype(np.float32)

    return X, y


def load_image_folder(image_path, label_path): ...

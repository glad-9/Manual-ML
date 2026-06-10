import numpy as np
import pandas as pd
import idx2numpy


def load_csv(file):
    df = pd.read_csv(file)
    return df


def load_idx_to_csv(file):
    ndarray_data = idx2numpy.convert_from_file(file)
    num_images, height, width = ndarray_data.shape
    flattened_data = ndarray_data.reshape(num_images, height * width)

    df = pd.DataFrame(flattened_data)
    return df


def load_idx(file):
    with open(file, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF  # last byte encodes number of dimensions

        dims = tuple(int.from_bytes(f.read(4), "big") for _ in range(ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)

    return data.reshape(dims)

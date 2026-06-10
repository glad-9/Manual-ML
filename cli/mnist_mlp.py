import numpy as np
import idx2numpy
from data_processing.split import split
from data_processing.tabular.tabulardataset import TabularDataset
from data_processing.tabular.normalizer import Normalizer


def run_mnist_mlp():
    X = idx2numpy.convert_from_file("datasets/raw/mnist/train-images.idx3-ubyte")
    y = idx2numpy.convert_from_file("datasets/raw/mnist/train-labels.idx1-ubyte")

    X = X.reshape(len(X), -1).astype(np.float32)  # (60000, 784)
    y = y.astype(np.float32)  # (60000,) sparse ints

    # one-hot here, before dataset creation
    def one_hot(labels, n_classes=10):
        out = np.zeros((len(labels), n_classes), dtype=np.float32)
        out[np.arange(len(labels)), labels.astype(int)] = 1.0
        return out

    y = one_hot(y)  # (60000, 10)

    raw_dataset = TabularDataset(X, y)
    train_set, val_set, test_set = split(raw_dataset, 0.8, 0.1)

    X_train_raw, y_train = train_set.get_all()
    X_val_raw, y_val = val_set.get_all()
    X_test_raw, y_test = test_set.get_all()

    normalizer = Normalizer()
    normalizer.fit(X_train_raw)

    train_set = TabularDataset(X_train_raw, y_train, transforms=[normalizer])
    val_set = TabularDataset(X_val_raw, y_val, transforms=[normalizer])
    test_set = TabularDataset(X_test_raw, y_test, transforms=[normalizer])

    return {
        "train": train_set,
        "cv": val_set,
        "test": test_set,
        "n_features": X.shape[1],
    }

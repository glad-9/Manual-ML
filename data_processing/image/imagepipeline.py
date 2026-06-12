import numpy as np

from data_processing.base import Pipeline

from data_processing.image.imagedataset import ImageDataset
from data_processing.split import split
from data_processing.image.imagenormalizer import ImageNormalizer
from data_processing.loaders import load_idx, load_image_folder


class ImagePipeline(Pipeline):
    def __init__(
        self, images_path, labels_path, train_ratio=0.8, val_ratio=0.1, normalize=True
    ):
        self.images = images_path
        self.labels = labels_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.normalize = normalize

    def run(self, idx=False, one_hot=True):
        # 1. Load
        if idx:
            X, y = load_idx(self.images, self.labels)
        else:
            X, y = load_image_folder(self.images, self.labels)

        # 2. One-hot labels
        def one_hot(labels, n_classes=10):
            out = np.zeros((len(labels), n_classes), dtype=np.float32)
            out[np.arange(len(labels)), labels.astype(int)] = 1.0
            return out

        y = one_hot(y, 10)

        raw_dataset = ImageDataset(X, y)
        train_set, val_set, test_set = split(
            raw_dataset, self.train_ratio, self.val_ratio
        )

        X_train, y_train = train_set.get_all()
        X_val, y_val = val_set.get_all()
        X_test, y_test = test_set.get_all()

        if self.normalize:
            normalizer = ImageNormalizer()
            normalizer.fit(X_train)
            train_set = ImageDataset(X_train, y_train, transforms=[normalizer])
            val_set = ImageDataset(X_val, y_val, transforms=[normalizer])
            test_set = ImageDataset(X_test, y_test, transforms=[normalizer])

        return {
            "train": train_set,
            "cv": val_set,
            "test": test_set,
        }

import os
import sys

# Calculate the parent directory (where nn and data_processing live)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import numpy as np

from nn.layers.recurrent.recurrent import Recurrent
from nn.layers.linear import Linear
from nn.initializers.xavier import Xavier

from data_processing.dataloader import DataLoader

from nn.network import Network
from nn.loss.mse import MSE
from nn.optim.adam import Adam


def generate_sine_dataset(
    n_points=2000,
    window_size=10,
    train_ratio=0.8,
    val_ratio=0.1,
    noise_std=0.0,
    seed=69,
):
    rng = np.random.default_rng(seed)

    t = np.linspace(0, 50 * np.pi, n_points)
    wave = np.sin(t).astype(np.float32)

    if noise_std > 0:
        wave = wave + rng.normal(0, noise_std, size=wave.shape).astype(np.float32)

    n_windows = n_points - window_size
    X = np.zeros((n_windows, window_size, 1), dtype=np.float32)
    y = np.zeros((n_windows, 1), dtype=np.float32)

    for i in range(n_windows):
        X[i, :, 0] = wave[i : i + window_size]
        y[i, 0] = wave[i + window_size]

    # Sequential split — no shuffling, since shuffling would leak future
    # points into the training set for a time series.
    train_end = int(n_windows * train_ratio)
    val_end = train_end + int(n_windows * val_ratio)

    return {
        "train": (X[:train_end], y[:train_end]),
        "cv": (X[train_end:val_end], y[train_end:val_end]),
        "test": (X[val_end:], y[val_end:]),
    }


data = generate_sine_dataset(noise_std=0.2, window_size=20)
X_train, y_train = data["train"]
X_val, y_val = data["cv"]
X_test, y_test = data["test"]

layers = [
    Recurrent(input_size=1, hidden_size=16, initializer=Xavier()),
    Linear(input_size=16, output_size=1, initializer=Xavier()),
]

dataloader = DataLoader(batch_size=256)
# dataloader.enabled = False

model = Network(layers, MSE, Adam(lr=0.01), dataloader)
train_cost, cv_cost = model.fit(
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    max_epochs=200,
    target_val_loss=0,
)

results, y_hat, y_true = model.evaluate(X_test, y_test)
print(
    f"---------------------------\nFinal Costs:\nFinal Train Cost: {train_cost[-1]}\nFinal Best CV Cost: {model.best_val_loss}"
)
print(
    f"---------------------------\nTest Set Results:\nTest Cost: {results['loss']}\nTest Accuracy: {results['accuracy']}"
)

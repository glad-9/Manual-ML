import numpy as np
import os
import pickle

from nn.layers.base import Layer
from core.tensor import Tensor


class Network:
    def __init__(self, layers, loss, optimizer, batcher):
        self.layers = layers  # List of Layer objects
        self.loss = loss()  # Loss object containing cost function and cost derivative
        self.optimizer = optimizer  # Pre-initialized optimizer
        self.batcher = batcher  # Pre-initialized Batcher object

        self.best_model_state = None
        self.best_val_loss = float("inf")

    def set_training_mode(self, mode: bool):
        Layer.set_training(mode)

    def forward_prop(self, X):
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
            # print(type(layer).__name__, activation.data.shape)

        return activation

    def compute_loss(self, X, y):
        y = y if isinstance(y, Tensor) else Tensor(y)
        y_hat = self.forward_prop(X)

        return self.loss.forward(y_hat, y)

    def backward_prop(self, X_batch, y_batch):
        loss = self.compute_loss(X_batch, y_batch)
        loss.backward()

        return loss.data

    def fit(
        self,
        train_data,
        max_epochs=1000,
        patience=20,
        val_data=None,
        save_path=None,
        target_val_loss=0.1,
    ):
        Layer.set_training(True)
        X_train, y_train = train_data
        train_history = []
        val_history = []
        patience_counter = 0

        batching = self.batcher.enabled

        for epoch in range(max_epochs):
            self.set_training_mode(True)
            epoch_losses = []

            if batching:
                for X_batch, y_batch in self.batcher.get_batch(X_train, y_train):
                    batch_cost = self.backward_prop(X_batch, y_batch)
                    epoch_losses.append(batch_cost)
                    self.optimizer.step(self.layers)
            else:
                batch_cost = self.backward_prop(X_train, y_train)
                self.optimizer.step(self.layers)
                epoch_losses.append(batch_cost)

            self.set_training_mode(False)
            total_train_loss = self.compute_loss(X_train, y_train).data.get()
            train_history.append(total_train_loss)

            print(
                f"Epoch: {epoch}\n"
                f"Full Train Loss: {total_train_loss}\n"
                f"Batch Train Loss: {epoch_losses[-1] if batching else 'NA'}"
            )

            if val_data is not None:
                X_val, y_val = val_data
                total_val_loss = self.compute_loss(X_val, y_val).data.get()
                val_history.append(total_val_loss)
                print(f"Val Cost: {total_val_loss}\n")

                if total_val_loss < self.best_val_loss:
                    self.best_val_loss = total_val_loss
                    patience_counter = 0

                    # Save best model
                    self.best_model_state = self.save_model()

                    if self.best_val_loss < target_val_loss:
                        print(
                            ""
                            f"---------------------------\nTarget CV Loss Achieved: {self.best_val_loss:.4f} < {target_val_loss}\nEarly stopping at epoch {epoch}"
                        )
                        break
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f"---------------------------\nMax Patience Reached: {patience}\nEarly stopping at epoch {epoch}"
                        )
                        break

        # Load best model at the end
        if self.best_model_state is not None:
            self.load_model(self.best_model_state)

        if save_path:
            self.save_model(save_path)

        return train_history, val_history

    def save_model(self, path=None):
        state = [layer.save_state() for layer in self.layers]

        if path:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
        return state

    def load_model(self, state=None, path=None):
        if state is None:
            state = []

        if path:
            with open(path, "rb") as f:
                state = pickle.load(f)
        for layer, layer_state in zip(self.layers, state):
            layer.load_state(layer_state)

    def predict(self, X):
        return self.forward_prop(X)

    def evaluate(self, X, y):
        self.set_training_mode(False)
        loss = self.compute_loss(X, y).data
        y_hat = self.predict(X)

        y_hat_np = y_hat.data.get()
        y_np = np.array(y.data if isinstance(y, Tensor) else y)
        accuracy = np.mean(y_hat_np.argmax(axis=1) == y_np.argmax(axis=1))
        return {"loss": loss, "accuracy": accuracy}, y_hat, y

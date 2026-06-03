import numpy as np
import os
import pickle

from nn.layers.base import Layer
from core.tensor import Tensor


class Network:
    def __init__(self, layers, loss, optimizer, batcher):
        self.layers = layers  # List of Layer objects
        self.loss = loss() # Loss object containing cost function and cost derivative
        self.optimizer = optimizer # Pre-initialized optimizer
        self.batcher = batcher # Pre-initialized Batcher object

        self.best_model_state = None
        self.best_val_cost = float('inf')

    def set_training_mode(self, mode:bool):
        Layer.set_training(mode)
    
    def forward_prop(self, X):
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        
        return activation

    def compute_loss(self, X, y):
        y = y if isinstance(y, Tensor) else Tensor(y)
        y_hat = self.forward_prop(X)
        
        return self.loss.forward(y_hat, y)
        

    def backward_prop(self, X_batch, y_batch):
        loss = self.compute_loss(X_batch, y_batch)
        loss.backward()
        
        return loss.data

    def fit(self, train_data, iterations=10000, patience=20, val_data=None, save_path=None):
        Layer.set_training(True)
        X_train, y_train = train_data
        train_cost_history = []
        patience_counter = 0
        
        for i in range(iterations):
            self.set_training_mode(True)
            epoch_costs = []

            if self.batcher.enabled:
                for X_batch, y_batch in self.batcher.get_batch(X_train, y_train):
                    batch_cost = self.backward_prop(X_batch, y_batch)
                    epoch_costs.append(batch_cost)

                    self.optimizer.step(self.layers)
            else:
                batch_cost = self.backward_prop(X_train, y_train)
                self.optimizer.step(self.layers)
            
            self.set_training_mode(False)

            full_train_cost = self.compute_loss(X_train, y_train).data
            train_cost_history.append(full_train_cost)

            if i % 100 == 0:
                print(f"Epoch: {i/100}\nFull Batch Train Cost: {train_cost_history[i]}\nBatch Train Cost: {epoch_costs[-1]}")

                if val_data is not None:
                    X_val, y_val = val_data
                    val_cost = self.compute_loss(X_val, y_val).data
                    print(f" Val Cost: {val_cost}")

                    if val_cost < self.best_val_cost:
                        self.best_val_cost = val_cost
                        patience_counter = 0

                        # Save best model 
                        self.best_model_state = self.save_model()

                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {i//100}")

                            # Load best model at the end
                            self.load_model(self.best_model_state)
                            break

        if self.best_model_state is not None:
            self.load_model(self.best_model_state)

        if save_path:
            self.save_model(save_path)

        return train_cost_history[-1], self.best_val_cost

    def save_model(self, path=None):
        state = [layer.save_state() for layer in self.layers]

        if path:
            if os.path.dirname(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path,"wb") as f:
                pickle.dump(state,f)
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
        accuracy = np.mean((y_hat.data >= 0.5) == y.data)
        return {"loss": loss, "accuracy": accuracy}
    

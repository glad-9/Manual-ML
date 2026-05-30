import numpy as np
import os
import pickle


class Network:
    def __init__(self, layers, cost, optimizer, batcher, X, y):
        self.X = X  # Inputs - (examples, features)
        self.y = y  # Labels - (examples, 1)
        self.layers = layers  # List of Layer objects
        self.cost = cost() # Cost object containing cost function and cost derivative
        self.optimizer = optimizer # Pre-initialized optimizer
        self.batcher = batcher # Pre-initialized Batcher object

        self.best_model_state = None
        self.best_val_cost = float('inf')
    
    def forward_prop(self, X=None):
        if X is None:
            X = self.X
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        
        return activation

    def compute_cost(self, X=None, y=None):
        if X is None:
            X = self.X
        
        if y is None:
            y = self.y

        y_hat = self.forward_prop(X)
        cost_value = self.cost.forward(y_hat, y) # cost
        dA = self.cost.backward(y_hat, y) # cost deriv

        return cost_value, dA

    def backward_prop(self, X_batch, y_batch, lambda_reg=0.01):
        cost, dA = self.compute_cost(X_batch, y_batch)

        for layer in reversed(self.layers):
            dA = layer.backward(dA, lambda_reg)

        return cost

    def fit(self, lr=0.01, lambda_reg=0.01, patience=20, iterations=5000, val_data=None, save_path=None):
        train_cost_history = []
        patience_counter = 0
        
        for i in range(iterations):
            epoch_costs = []

            if self.batcher.enabled:
                for X_batch, y_batch in self.batcher.get_batch(self.X, self.y):
                    batch_cost = self.backward_prop(X_batch, y_batch, lambda_reg)
                    epoch_costs.append(batch_cost)

                    self.optimizer.step(self.layers)

            full_train_cost, _ = self.compute_cost(self.X, self.y)
            train_cost_history.append(full_train_cost)

            if i % 100 == 0:
                print(f"Epoch: {i/100}\n Train Cost: {train_cost_history[i]}")

                if val_data is not None:
                    X_val, y_val = val_data
                    val_cost = self.compute_cost(X_val, y_val)[0]
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

    def predict(self, X=None):
        return self.forward_prop(X)

    def evaluate(self, X, y):
        cost, _ = self.compute_cost(X, y)
        y_hat = self.predict(X)
        accuracy = np.mean((y_hat >= 0.5) == y)
        return {"cost": cost, "accuracy": accuracy}
    

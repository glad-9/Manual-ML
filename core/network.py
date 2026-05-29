import numpy as np
import pickle


class Network:
    def __init__(self, layers, cost, X, y):
        self.X = X  # Inputs - (examples, features)
        self.y = y  # Labels - (examples, 1)
        self.layers = layers  # List of Layer objects
        self.cost_func = cost.forward # Cost/Loss function (forward attribute in Cost object)
        self.cost_deriv = cost.backward  # Cost/Loss derivative (backward attribute in Cost object)

        self.best_weights = None
        self.best_val_cost = float('inf')
    
    def forward_prop(self, X=None):
        if X is None:
            X = self.X
        activation = X
        for layer in self.layers:
            activation = layer.activate(activation)
        
        return activation

    def compute_cost(self, X=None, y=None):
        if X is None:
            X = self.X
        
        if y is None:
            y = self.y

        y_hat = self.forward_prop(X)
        cost_value = self.cost_func(y_hat, y) # cost
        dA = self.cost_deriv(y_hat, y) # cost deriv

        return cost_value, dA

    def backward_prop(self, lambda_reg=0.01):
        reversed_layers = reversed(self.layers)
        cost, dA = self.compute_cost()
        for layer in reversed_layers:
            dA = layer.compute_gradients(dA, lambda_reg)

        return cost

    def train(self, lr=0.01, lambda_reg=0.01, patience=20, iterations=5000, val_data=None):
        cost_history = []
        patience_counter = 0
        
        for i in range(iterations):
            cost = self.backward_prop(lambda_reg)
            for layer in self.layers:
                layer.update_params(lr)

            cost_history.append(cost)

            if i % 100 == 0:
                print(f"Epoch: {i/100}\n Train Cost: {cost_history[i]}")

                X_val, y_val = val_data
                if val_data is not None:
                    val_cost = self.compute_cost(X_val, y_val)[0]
                    print(f" Val Cost: {val_cost}")

                    if val_cost < self.best_val_cost:
                        self.best_val_cost = val_cost
                        patience_counter = 0

                        # Save best model 
                        best_model_state = self.save_model()

                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {i}")

                            # Load best model at the end
                            self.load_model(best_model_state)
                            break

        return cost_history[-1]

    def save_model(self, path=None):
        return [layer.save_state() for layer in self.layers]

    def load_model(self, state):
        for layer, layer_state in zip(self.layers, state):
            layer.load_state(layer_state)

    

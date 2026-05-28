import numpy as np


class Network:
    def __init__(self, layers, cost, X, y):
        self.X = X  # Inputs - (examples, features)
        self.y = y  # Labels - (examples, 1)
        self.layers = layers  # List of Layer objects
        self.cost_func = cost.forward # Cost/Loss function (forward attribute in Cost object)
        self.cost_deriv = cost.backward  # Cost/Loss derivative (backward attribute in Cost object)
    
    def forward_prop(self):
        activation = self.X
        for layer in self.layers:
            activation = layer.activate(activation)
        
        return activation

    def compute_cost(self):
        y_hat = self.forward_prop()
        cost_value = self.cost_func(y_hat, self.y) # cost
        dA = self.cost_deriv(y_hat, self.y) # cost deriv

        return cost_value, dA

    def backward_prop(self):
        reversed_layers = reversed(self.layers)
        cost, dA = self.compute_cost()
        for layer in reversed_layers:
            dA = layer.compute_gradients(dA)

        return cost

    def train(self, lr, iterations):
        cost_history = []
        
        for i in range(iterations):
            cost = self.backward_prop()
            for layer in self.layers:
                layer.update_params(lr)

            cost_history.append(cost)

            if i % 100 == 0:
                print(f"Epoch: {i/100}\n Cost: {cost_history[i]}")

        return np.mean(cost_history)

    

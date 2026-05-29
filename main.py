import numpy as np

from data_processing.pipeline import pipeline
from core.layers.dense import Dense
from core.activations import linear_activation, linear_deriv, relu_activation, relu_deriv, sigmoid_activation, sigmoid_deriv, Activation
from core.losses import bce, bce_deriv, Loss
from core.network import Network

def main():
    dataset_path = "datasets/raw/diabetes.csv"
    subsets = pipeline(dataset_path, "Outcome", 0.8, 0.1)

    X_train, y_train = subsets["train"]
    X_val, y_val = subsets["cv"]
    X_test, y_test = subsets["test"]

    feature_count = X_train.shape[1]

    # Activations
    linear = Activation(linear_activation, linear_deriv)
    relu = Activation(relu_activation, relu_deriv)
    sigmoid = Activation(sigmoid_activation, sigmoid_deriv)

    layer_1 = Dense(input_size=feature_count, count=32, activation=relu)
    layer_2 = Dense(input_size=32, count=16, activation=relu)
    layer_3 = Dense(input_size=16, count=8, activation=sigmoid)
    layer_4 = Dense(input_size=8, count=1, activation=sigmoid)

    layers = [layer_1, layer_2, layer_3, layer_4]

    # Cost
    loss = Loss(bce, bce_deriv)

    network = Network(layers, loss, X_train, y_train)
    final_train_cost = network.train(lr=0.005, lambda_reg=0.05, iterations=50000, val_data=(X_val,y_val))

    cv_cost = network.compute_cost(X_val,y_val)[0]

    print(f"Training Final Cost: {final_train_cost}\n CV Cost: {cv_cost}")



if __name__ == '__main__':
    main()

import numpy as np

from data_processing.pipeline import pipeline
from core.layers.dense import Dense
from core.activations import reLU, reLU_deriv, sigmoid, sigmoid_deriv, Activation
from core.losses import bce, bce_deriv, Loss
from core.network import Network

def main():
    dataset_path = "datasets/raw/diabetes.csv"
    subsets = pipeline(dataset_path, "Outcome")

    X_train, y_train = subsets["train"]
    X_val, y_val = subsets["cv"]
    X_test, y_test = subsets["test"]

    feature_count = X_train.shape[1]

    # Activations
    relu = Activation(reLU, reLU_deriv)
    sigmoid_activation = Activation(sigmoid, sigmoid_deriv)

    layer_1 = Dense(input_size=feature_count, count=64, activation=relu)
    layer_2 = Dense(input_size=64, count=32, activation=relu)
    layer_3 = Dense(input_size=32, count=16, activation=relu)
    layer_4 = Dense(input_size=16, count=8, activation=relu)
    layer_5 = Dense(input_size=8, count=1, activation=sigmoid_activation)

    layers = [layer_1, layer_2, layer_3, layer_4, layer_5]

    # layer_1 = Dense(input_size = feature_count, count = 32, activation = relu)
    # layer_2 = Dense(input_size = 32, count=16, activation=relu)
    # layer_3 = Dense(input_size = 16, count = 8, activation=relu)
    # layer_4 = Dense(input_size = 8, count = 1, activation=sigmoid_activation)

    # layers = [layer_1, layer_2, layer_3, layer_4]

    # Cost
    loss = Loss(bce, bce_deriv)

    network = Network(layers, loss, X_train, y_train)
    final_train_cost = network.train(0.01, 5000)

    print(f"Training Final Cost: {final_train_cost}")


if __name__ == '__main__':
    main()

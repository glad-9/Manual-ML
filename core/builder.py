import yaml
from core.layers.dense import Dense
from core.activations import Linear, ReLU, Sigmoid
from core.losses import BCE, MSE
from core.network import Network

ACTIVATIONS = {
    "linear": Linear,
    "relu":  ReLU,
    "sigmoid": Sigmoid,
}

LOSSES = {
    "mse":MSE,
    "bce":BCE,
}

def build_network(config_path, X_train, y_train):
    with open (config_path) as f:
        config = yaml.safe_load(f)

    feature_count = X_train.shape[1]
    prev_size = feature_count

    layers = []
    for layer_cfg in config["layers"]:
        activation = ACTIVATIONS[layer_cfg["activation"]]
        layer = Dense(
            input_size=prev_size,
            count=layer_cfg["count"],
            activation=activation,
        )
        prev_size = layer_cfg["count"] # next layer's input size
        layers.append(layer)

    loss = LOSSES[config["training"]["loss"]]

    return Network(layers, loss, X_train, y_train)
import yaml

from core.layers.dense import Dense
from core.network import Network

from core.activations import Linear, ReLU, Sigmoid

from core.losses import BCE, MSE

from core.optimizers.sgd import SGD
from core.optimizers.adam import Adam

from data_processing.batching import Batcher


ACTIVATIONS = {
    "linear": Linear,
    "relu":  ReLU,
    "sigmoid": Sigmoid,
}

LOSSES = {
    "mse":MSE,
    "bce":BCE,
}

OPTIMIZERS = {
    "sgd": SGD,
    "adam": Adam,
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

    optimizer = OPTIMIZERS[config["training"]["optimizer"]]
    optimizer_instance = optimizer(lr=config["training"]["lr"])

    batch_cfg = config["training"].get("batching", {})
    batcher = Batcher(
        method=batch_cfg.get("method", "standard"),
        batch_size=batch_cfg.get("batch_size", 32),
        drop_last=batch_cfg.get("drop_last", False)
    )

    return Network(layers, loss, optimizer_instance, batcher, X_train, y_train)
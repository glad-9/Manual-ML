import yaml

from core.network import Network

from core.initializers.he import He
from core.initializers.xavier import Xavier

from core.layers.linear import Linear

from core.activations.relu import ReLU
from core.activations.sigmoid import Sigmoid
from core.activations.tanh import Tanh

from core.losses.bce import BCE
from core.losses.mse import MSE

from core.optimizers.sgd import SGD
from core.optimizers.adagrad import Adagrad
from core.optimizers.momentum import Momentum
from core.optimizers.rmsprop import RMSprop
from core.optimizers.adam import Adam

from data_processing.batching import Batcher

INITIALIZERS = {
    "he": He,
    "xavier": Xavier,
}

LAYERS = {
    "linear": Linear,
    "relu":  ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
}

LOSSES = {
    "mse":MSE,
    "bce":BCE,
}

OPTIMIZERS = {
    "sgd": SGD,
    "adagrad": Adagrad,
    "momentum": Momentum,
    "rmsprop": RMSprop,
    "adam": Adam,
}

def build_network(config_path, feature_count):
    with open (config_path) as f:
        config = yaml.safe_load(f)

    prev_size = feature_count

    layers = []
    for layer_cfg in config["layers"]:
        initializer_key = layer_cfg.get("initializer", "he")
        initializer = INITIALIZERS[initializer_key]()

        layer_type = layer_cfg["type"]

        if layer_type == "linear":
            layer = LAYERS[layer_type](
                input_size=prev_size,
                output_size=layer_cfg["output_size"],
                initializer=initializer
            )
            prev_size = layer_cfg["output_size"] # next layer's input size
        else:
            layer = LAYERS[layer_type]()

        layers.append(layer)

    loss = LOSSES[config["training"]["loss"]]

    optimizer_cfg = config["training"].get("optimizer", {})
    optimizer = OPTIMIZERS[optimizer_cfg.pop("type")]
    optimizer_instance = optimizer(**optimizer_cfg)

    batch_cfg = config["training"].get("batching", {})
    batcher = Batcher(
        method=batch_cfg.get("method", "standard"),
        batch_size=batch_cfg.get("batch_size", 32),
        drop_last=batch_cfg.get("drop_last", True),
        shuffle=batch_cfg.get("shuffle", True)
    )
    if not batch_cfg.get("enabled"):
        batcher.enabled = False

    return Network(layers, loss, optimizer_instance, batcher)
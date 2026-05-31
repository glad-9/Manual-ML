import yaml

from core.network import Network

from core.initializers.he import He
from core.initializers.xavier import Xavier

from core.layers.linear import Linear
from core.layers.dropout import Dropout
from core.layers.batchnorm import BatchNorm

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
    "dropout": Dropout,
    "batchnorm": BatchNorm,

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

def build_layer(layer_cfg, prev_size):
    layer_type = layer_cfg["type"]

    if layer_type == "linear":
        initializer = INITIALIZERS[layer_cfg.get("initializer", "he")]()
        layer = Linear(prev_size, layer_cfg["output_size"], initializer=initializer)
        return layer, layer_cfg["output_size"]

    elif layer_type == "batchnorm":
        return BatchNorm(prev_size), prev_size

    elif layer_type == "dropout":
        return Dropout(layer_cfg["rate"]), prev_size

    elif layer_type in LAYERS:
        return LAYERS[layer_type](), prev_size


def build_network(config_path, feature_count):
    with open (config_path) as f:
        config = yaml.safe_load(f)

    prev_size = feature_count

    layers = []
    for layer_cfg in config["layers"]:
        layer, prev_size = build_layer(layer_cfg, prev_size)
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
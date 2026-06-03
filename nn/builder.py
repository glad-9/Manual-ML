import yaml

from nn.network import Network

from nn.initializers.he import He
from nn.initializers.xavier import Xavier

from nn.layers.linear import Linear
from nn.layers.dropout import Dropout
from nn.layers.batchnorm import BatchNorm

from nn.activations.relu import ReLU
from nn.activations.sigmoid import Sigmoid
from nn.activations.tanh import Tanh

from nn.loss.bce import BCE
from nn.loss.mse import MSE

from nn.optim.sgd import SGD
from nn.optim.adagrad import Adagrad
from nn.optim.momentum import Momentum
from nn.optim.rmsprop import RMSprop
from nn.optim.adam import Adam

from data_processing.dataloader import DataLoader

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
    dataloader = DataLoader(
        method=batch_cfg.get("method", "standard"),
        batch_size=batch_cfg.get("batch_size", 32),
        drop_last=batch_cfg.get("drop_last", True),
        shuffle=batch_cfg.get("shuffle", True)
    )
    if not batch_cfg.get("enabled"):
        batcher.enabled = False

    return Network(layers, loss, optimizer_instance, dataloader)
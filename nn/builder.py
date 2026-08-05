import yaml

from core.tensor import Tensor
from nn.network import Network

from nn.initializers.he import He
from nn.initializers.xavier import Xavier

from nn.layers.linear import Linear

from nn.layers.conv.conv2d import Conv2D
from nn.layers.conv.flatten import Flatten
from nn.layers.conv.maxpool2d import MaxPool2D

from nn.layers.regularization.dropout import Dropout
from nn.layers.regularization.batchnorm import BatchNorm

from nn.layers.recurrent.recurrent import Recurrent

from nn.layers.activations.relu import ReLU
from nn.layers.activations.sigmoid import Sigmoid
from nn.layers.activations.tanh import Tanh
from nn.layers.activations.softmax import Softmax

from nn.loss.bce import BCE
from nn.loss.mse import MSE
from nn.loss.ce import CE

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
    "conv2d": Conv2D,
    "maxpool2d": MaxPool2D,
    "flatten": Flatten,
    "recurrent": Recurrent,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}

LOSSES = {
    "mse": MSE,
    "bce": BCE,
    "ce": CE,
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

    if layer_type == "linear" or layer_type == "recurrent":
        layer_instance = LAYERS[layer_cfg.get("type")]
        initializer = INITIALIZERS[layer_cfg.get("initializer", "he")]()
        input_size = layer_cfg.get("input_size", None)
        layer = Linear(
            input_size=input_size if input_size is not None else prev_size,
            output_size=layer_cfg["output_size"],
            initializer=initializer,
        )
        return layer, layer_cfg["output_size"]

    elif layer_type == "batchnorm":
        return BatchNorm(prev_size), prev_size

    elif layer_type == "dropout":
        return Dropout(layer_cfg["rate"]), prev_size

    elif layer_type == "conv2d":
        initializer = INITIALIZERS[layer_cfg.get("initializer", "he")]()
        k_size = layer_cfg["kernel_size"]
        k_size = tuple(k_size) if isinstance(k_size, list) else (k_size, k_size)
        layer = Conv2D(
            in_channels=prev_size,
            out_channels=layer_cfg["out_channels"],
            k_size=k_size,
            initializer=initializer,
            stride=layer_cfg.get("stride", 1),
            pad=layer_cfg.get("pad", False),
        )
        return layer, layer_cfg["out_channels"]

    elif layer_type == "maxpool2d":
        return MaxPool2D(
            pool_size=layer_cfg["pool_size"],
            stride=layer_cfg.get("stride", None),
        ), prev_size

    elif layer_type in LAYERS:
        return LAYERS[layer_type](), prev_size


def build_network(config_path, input_size):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    prev_size = input_size

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
        shuffle=batch_cfg.get("shuffle", True),
    )
    if not batch_cfg.get("enabled"):
        dataloader.enabled = False

    return Network(layers, loss, optimizer_instance, dataloader)

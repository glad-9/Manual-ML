import numpy as np

from layers.base import Layer
from core.tensor import Tensor


class Conv2D(Layer):
    def __init__(
        self, in_channels, out_channels, k_size, initializer, stride=1, pad=False
    ):
        self.kH, self.kW = k_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pad = (k_size - 1) // 2 if pad else 0

        self.shape = (self.out_channels, self.in_channels, self.kH, self.kW)
        self.W = Tensor(
            initializer.initialize(self.shape), requires_grad=True, requires_reg=True
        )
        self.b = Tensor(
            np.zeros((1, out_channels, 1, 1)).astype("float32"), requires_grad=True
        )

    """
    
    """

    def _im2col(self, X, kH, kW, stride):
        # Extracting dimensions from input
        n, c, h, w = X.shape

        # Calculating the number of patches we can get with our kernel
        h_out = ((h - kH) // stride) + 1
        w_out = ((w - kW) // stride) + 1

        # Flattening X into (total images in batch (depth), total number of patches (rows), dot products after multiplying convolving image values with kernel weights(cols))
        col = np.zeros((n, h_out * w_out, c * kH * kW))

        # Iterating over every kernel position to fill out the zero matrix
        for i in range(h_out):
            for j in range(w_out):
                # Top-left corner of the patch at position (i, j)
                h_start = i * stride
                w_start = j * stride

                # Extract patches across all N images and all C channels
                # Patch shape: (n, c, kH, kW)
                patch = X[:, :, h_start : h_start + kH, w_start : w_start + kW]

                # Flatten patch and insert as a row
                col[:, i * w_out + j, :] = patch.reshape(n, -1)

        return col

    def forward(self, X):
        X = X if isinstance(X, Tensor) else Tensor(X)
        if self.pad != 0:
            X = X.pad2d(self.pad)

        n, _, h, w = X.data.shape

        h_out = ((h - self.kH) // self.stride) + 1
        w_out = ((w - self.kW) // self.stride) + 1

        col = Tensor(self._im2col(X.data, self.kH, self.kW, self.stride))
        w_flat = self.W.reshape(self.out_channels, -1).transpose(1, 0)

        out = col @ w_flat
        out = out.reshape(n, h_out, w_out, self.out_channels)  # (N, H', W', K)
        out = out.transpose(0, 3, 1, 2)
        out += self.b

        return out

    def get_params(self):
        return [self.W, self.b]

    def save_state(self):
        return {"W": self.W.data.copy(), "b": self.b.data.copy()}

    def load_state(self, state):
        self.W = Tensor(state["W"].copy(), requires_grad=True, requires_reg=True)
        self.b = Tensor(state["b"].copy(), requires_grad=True)

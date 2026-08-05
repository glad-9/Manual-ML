import numpy as np

from core.tensor import Tensor
from nn.layers.base import Layer


class Conv2D(Layer):
    """
    2-D Convolutional Layer.

    A layer consisting of a number of 'filters/kernels' that are analagous to neurons in a linear layer.
    Convolutional layers are primarily used when working with visual data. They are specifically
    designed to exploit spacial relationships within visual data, significantly reducing
    parameter counts, and automatically learning hierarchical features (edges, colors, complex shapes).

    Conv layers apply filters/kernels to small, local patches of an image at a time rather than
    looking at the entire image at once. Kernels are a set of weights in a grid-like fashion that
    can be slid across the whole image thereby significantly reducing the total number of
    learnable parameters compared to a fully connected network where each pixel in an image
    is mapped to an output.

    Because conv layers are stacked, each filter/kernel within the layer can learn to 'look'
    for different features (edges, curves, colors, etc.). As the network gets deeper, the
    kernels get more sophisticated and learn to 'see' emerging hierarchical features (faces, objects, etc.)

    Lastly, because these layers avoid flattening the entire image into a 1D vector at the start,
    convolution retains spacial relationships between neighboring pixels.


    Attributes
    ----------
    in_channels : int
        Number of channels received in input by this layer.
    out_channels : int
        Number of channels in the output image (AKA no. of kernels/filters for this layer).
    initializer : `Initializer`
        An instance of the `Initializer` object to initialize the weights.
    kH : int
        Kernel height.
    kW : int
        Kernel width.
    stride : int
        Kernel stride.
        Default is 1.
    pad : int
        Padding size to be applied to a tensor.
    W : Tensor
        Weight tensor with gradient calculations and regularization enabled.
    b : Tensor
        Bias tensor with gradient calculations enabled.
    h_out : int
        Height of the convolved image.
    w_out : int
        Width of the convolved image.
    """

    def __init__(
        self, in_channels, out_channels, initializer, k_size, stride=1, pad=False
    ):
        """
        Initializes a Conv2D Layer instance.

        Parameters
        ----------
        in_channels : int
            Number of channels received in input by this layer.
        out_channels : int
            Number of channels in the output image (AKA no. of kernels/filters for this layer).
        initializer : `Initializer`
            An instance of the `Initializer` object to initialize the weights.
        k_size : tuple of int
            Tuple representing kernel dimensions.
        stride : int
            Kernel stride.
            Default is 1.
        pad : bool
            Whether to use valid (false) or same (true) padding.
            Default is False.
        """
        self.kH, self.kW = k_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.pad = (k_size[0] - 1) // 2 if pad else 0

        shape = (self.out_channels, self.in_channels, self.kH, self.kW)
        self.W = Tensor(
            initializer.initialize(shape), requires_grad=True, requires_reg=True
        )
        self.b = Tensor(
            np.zeros((1, out_channels, 1, 1)).astype("float32"), requires_grad=True
        )

    def forward(self, X):
        """
        Forward pass of a convolutional layer.

        Parameters
        ----------
        X : Tensor or array-like
            Input from previous layer.

        Returns
        -------
        Tensor
            A new convolved tensor with dimensions (n, out_channels, h_out, w_out).

        Notes
        -----
        Applies the im2col operation on the input Tensor to flatten it. The weight matrix is then flattened to support a matmul op with the flattened X.
        """
        X = X if isinstance(X, Tensor) else Tensor(X)
        if self.pad != 0:
            X = X.pad2d(self.pad)

        n, _, h, w = X.data.shape

        self.h_out = ((h - self.kH) // self.stride) + 1
        self.w_out = ((w - self.kW) // self.stride) + 1

        # Shape : (n, h_out * w_out, kh * kw * in_c)
        col = X.im2col(self.kH, self.kW, self.stride, self.h_out, self.w_out)

        # Shape : (kh * kw * in_c, out_c)
        w_flat = self.W.reshape(self.out_channels, -1).transpose(1, 0)

        out = col @ w_flat  # (n, h_out * w_out, out_c)
        out = out.reshape(
            n, self.h_out, self.w_out, self.out_channels
        )  # (n, h_out, w_out, out_c)
        out = out.transpose(0, 3, 1, 2)  # (n, out_c, h_out, w_out)
        out += self.b

        return out

    def get_params(self):
        """Return a list of trainable parameters."""
        return [self.W, self.b]

    def save_state(self):
        """Return a copy of the underlying weights and biases."""
        return {"W": self.W.data.copy(), "b": self.b.data.copy()}

    def load_state(self, state):
        """
        Load parameter data safely into the existing Tensor instances.

        Parameters
        ----------
        state : dict
            A dictionary containing "W" and "b" state arrays.
        """
        self.W = Tensor(state["W"].copy(), requires_grad=True, requires_reg=True)
        self.b = Tensor(state["b"].copy(), requires_grad=True)

import numpy as np

from core.tensor import Tensor
from .base import Layer


class Linear(Layer):
    """
    A fully-connected layer of neurons (represented as rows in the weight matrix).
    Generally followed by an activation layer (ReLU, Sigmoid, Softmax, etc.) for non-linearity.

    Attributes
    ----------
    input_size : int
        Size of the previous layer's output.
    output_size : int
        Output size of this layer (number of neurons).
    W : Tensor
        Weight tensor with gradient calculations and regularization enabled.
    b : Tensor
        Bias tensor with gradient calculations enabled.
    """

    def __init__(self, input_size, output_size, initializer):
        """
        Initialize a Linear Layer instance.

        Parameters
        ----------
        input_size : int
            Number of input features.
        output_size : int
            Activation size of this layer (number of neurons).
        initializer : `Initializer`
            An instance of the `Initializer` object to initialize the weights.
        """
        self.input_size = input_size
        self.output_size = output_size

        self.W = Tensor(
            initializer.initialize((input_size, output_size)),
            requires_grad=True,
            requires_reg=True,
        )
        self.b = Tensor(np.zeros((1, self.output_size)), requires_grad=True)

    def forward(self, X):
        """
        Parameters
        ----------
        X : Tensor or array-like
            Input used for this layer's forward.

        Returns
        -------
        Tensor
            A new tensor connected to the computation graph with the linear activation:
            Z = X @ W + b
        """
        X = X if isinstance(X, Tensor) else Tensor(X)
        return X @ self.W + self.b

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

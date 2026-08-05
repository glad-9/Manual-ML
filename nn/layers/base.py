from abc import ABC, abstractmethod
from core.tensor import Tensor


class Layer(ABC):
    """
    Abstract base class for layers.

    Subclasses must implement forward(), save_state(), load_state() to define a given layer type's activations and stored parameters.

    Attributes
    ----------
    training : bool
        Whether layers are currently in training or inference mode.
        This is a class attribute defined on `Layer` meaning toggling it affects every single layer.
        Default is True.

    Contract
    --------
    Subclasses must implement:

    forward(X):
        Must return a `Tensor` so the operation is tracked within the computation graph. Any layer that builds output data from raw NumPy/CuPy without routing it through Tensor ops disconnects the graph (see docs/debug-log/cnn_none_grad.md).

    get_params():
        Must return a list of all trainable parameters used within the layer even if empty. Used by the optimizer to calculate steps.

    save_state() / load_state():
        Must be inverses of each other - save_state() returns a dict that load_state() can consume to fully restore the layer's parameters.
    """

    training: bool = True

    @classmethod
    def set_training(cls, mode: bool):
        """Setter for `training` attribute."""
        cls.training = mode

    @classmethod
    def is_training(cls) -> bool:
        """Getter for `training` attribute."""
        return Layer.training

    @abstractmethod
    def forward(self, X):
        """
        Computes the output of this layer.
        Implemented by subclasses.

        Parameters
        ----------
        X : Tensor or array-like
            Input from the previous layer.

        Returns
        -------
        Tensor
            A new tensor with the forward-pass computations applied
        """
        return Tensor(...)

    @abstractmethod
    def get_params(self):
        """Return trainable parameters as a list."""
        return []

    @abstractmethod
    def save_state(self):
        """Return the current weights and biases of the layer as a dict."""
        return {}

    @abstractmethod
    def load_state(self, state):
        """Loads weights and biases from a state dict."""
        pass

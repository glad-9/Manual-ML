import numpy as np
from core.tensor import Tensor
from nn.layers.base import Layer


class BatchNorm(Layer):
    """
    Normalizes the inputs of the previous layer to have a mean of zero and a standard deviation of 1.
    Generally used BEFORE an activation.

    Facilitates faster training and stable convergence as the result of normalized inputs.

    Attributes
    ----------
    input_size : int
        Previous layer's output size.
    gamma : Tensor
        Learnable scale with gradient calculations enabled.
        Allows the network to learn different distributions.
    beta : Tensor
        Learnable shift with gradient calculations enabled.
        Allows the network to learn different distributions.
    momentum : float
        Determines how much of the previous mean/variance to preserve
    running_mean : Tensor
        Running average of means to be used in inference.
    running_var : Tensor
        Running average of variances to be used in inference.
    epsilon : float
        Small number to prevent division by zero.
    """

    def __init__(self, input_size, momentum=0.9, epsilon=1e-8):
        """
        Initializes a BatchNorm Layer instance.

        Parameters
        ----------
        input_size : int
            Previous layer's output size.
        momentum : float
            Determines how much of the previous mean/variance to preserve
        epsilon : float
            Small number to prevent division by zero.
        """
        self.gamma = Tensor(
            np.ones((1, input_size)).astype("float32"), requires_grad=True
        )  # Learnable scale
        self.beta = Tensor(
            np.zeros((1, input_size)).astype("float32"), requires_grad=True
        )  # Learnable shift
        self.momentum = momentum
        self.epsilon = epsilon

        # Used in inference
        self.running_mean = Tensor(np.zeros((1, input_size)).astype("float32"))
        self.running_var = Tensor(np.zeros((1, input_size)).astype("float32"))

    def forward(self, X):
        """
        Apples batch normalization to the outputs of the previous layer.

        Parameters
        ----------
        X : Tensor
            Previous layer's activation/output.

        Returns
        -------
        Tensor
            A new normalized activation tensor.

        Notes
        -----
        Calculates the running_mean and running_var during training that are then used in inference instead of dynamically calculating mean and variance.
        Scales and shifts the normalized outputs of the previous layer.
        """
        if self.is_training():
            # mean = (1/m) * sum(X)
            mean = X.mean()

            # var = (1/m) * sum((X - mean) ^ 2)
            var = X.var()

            X_norm = (X - mean) / (var + self.epsilon) ** 0.5

            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            return self.gamma * X_norm + self.beta

        else:
            X_norm = (X - self.running_mean) / (self.running_var + self.epsilon) ** 0.5
            return self.gamma * X_norm + self.beta

    def get_params(self):
        """Returns the list of trainable parameters for this layer (gamma and beta)."""
        return [self.gamma, self.beta]

    def save_state(self):
        """Return a copy of parameters used in training and inference."""
        return {
            "gamma": self.gamma.data.copy(),
            "beta": self.beta.data.copy(),
            "running_mean": self.running_mean.data.copy(),
            "running_var": self.running_var.data.copy(),
        }

    def load_state(self, state):
        """
        Load parameter data safely into the existing Tensor instances.

        Parameters
        ----------
        state : dict
            A dictionary containing "gamma", "beta", "running_mean", and "running_var" state arrays.
        """
        self.gamma = Tensor(state["gamma"].copy(), requires_grad=True)
        self.beta = Tensor(state["beta"].copy(), requires_grad=True)
        self.running_mean = Tensor(state["running_mean"].copy())
        self.running_var = Tensor(state["running_var"].copy())

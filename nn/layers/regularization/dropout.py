import numpy as np

from nn.layers.base import Layer


class Dropout(Layer):
    """
    Drops out a percentage of the previous layer's neurons during training.
    Generally applied AFTER an activation.

    Forces the layer's neurons to generalize rather than memorize the training data.

    Attributes
    ----------
    rate : float
        Probability of setting an element to zero (between 0 and 1).
    """

    def __init__(self, dropout_rate):
        """
        Initialize a Dropout Layer instance.

        Parameters
        ----------
        dropout_rate : float
            Probability of setting an element to zero (between 0 and 1).
        """
        self.rate = dropout_rate

    def forward(self, X):
        """
        Apply inverted dropout to the input tensor during training.

        Parameters
        ----------
        X : Tensor
            Activation from the previous layer.

        Returns
        -------
        Tensor
            The scaled tensor with randomly zeroed elements during training
            or the identical input tensor during eval mode.

        Notes
        -----
        Randomly generates a 'mask' that zeroes out the percentage of neurons based on this layer's rate.
        Following that, we scale the rest of the input via division by the complement of rate.
        """
        if self.is_training():
            mask = np.random.randn(*X.shape) > self.rate
            return X * mask / (1 - self.rate)
        return X

    def get_params(self):
        """Returns an empty list as dropout contains no learnable parameters."""
        return []

    def save_state(self):
        """Return an empty dict as dropout has no persistent weights."""
        return {}

    def load_state(self, state):
        """No-op because dropout has no structural state to load."""
        pass

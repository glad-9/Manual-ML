from nn.layers.base import Layer


class Flatten(Layer):
    """
    Flattens the output of a Conv2D or MaxPool2D layer to use within a Linear layer.
    """

    def forward(self, X):
        """
        Forward operation of a Flatten Layer.

        Parameters
        ----------
        X : Tensor
            Output of the previous layer.

        Returns
        -------
        Tensor
            A new flattened tensor that can be fed into a linear layer.
        """
        n = X.data.shape[0]
        return X.reshape(n, -1)

    def get_params(self):
        """Returns an empty list as flatten contains no learnable parameters."""
        return []

    def save_state(self):
        """Return an empty dict as flatten has no persistent weights."""
        return {}

    def load_state(self, state):
        """No-op because flatten has no structural state to load."""
        pass

from nn.layers.base import Layer


class MaxPool2D(Layer):
    """
    Max-Pooling Layer.

    This layer is often seen following a Conv layer (see nn/layers/conv/conv2d.py), and its
    primary function is to reduce the overall computation cost by decreasing spatial
    dimensionality of the convolved image by sliding a pooling 'filter' across the convolution
    and only choosing the maximum/most significant value within that filter (generally a 2x2 patch).
    """

    def __init__(self, pool_size, stride):
        """
        Initializes a MaxPool2D Layer instance.
        """
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        """Uses the tensor maxpooling operation to reduce the size of the convolved output of the previous layer."""
        return X.pool2d(self.pool_size, self.stride)

    def get_params(self):
        """Returns an empty list as pooling contains no learnable parameters."""
        return []

    def save_state(self):
        """Return an empty dict as pooling has no persistent weights."""
        return {}

    def load_state(self, state):
        """No-op because pooling has no structural state to load."""
        pass

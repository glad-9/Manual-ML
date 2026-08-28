import numpy as np

from core.tensor import Tensor
from nn.layers.base import Layer


class Recurrent(Layer):
    """
    Recurrent Layer.

    Recurrent layers process sequential data (text, time series, audio) by
    maintaining a hidden state that acts as short-term memory. At each state,
    the layer takes a new input and combines it with the previous hidden state,
    passing updated context forward through time.

    Attributes
    ----------
    input_size : int
        Number of features for a given timestep (e.g: 300-dimensional word embeddings per word in a sentence).
    hidden_size : int
        Size of the recurrent layer.
    W_xh : Tensor
        Weights that map a current timestep into the hidden space.
    W_hh : Tensor
        Weights that transform the previous hidden states before adding to the new input.
    b_h : Tensor
        Bias added to sum of the transformed input.
    """

    def __init__(self, input_size, hidden_size, initializer):
        """
        Initializes a Recurrent Layer instance.

        Parameters
        ----------
        input_size : int
            Size of features for a given timestep.
        hidden_size : int
            Size of the recurrent layer.
        initializer : `Initializer`
            An instance of the `Initializer` object to initialize the weights.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_xh = Tensor(
            initializer.initialize((input_size, hidden_size)),
            requires_grad=True,
            requires_reg=True,
        )
        self.W_hh = Tensor(
            initializer.initialize((hidden_size, hidden_size)),
            requires_grad=True,
            requires_reg=True,
        )
        self.b_h = Tensor(np.zeros((1, hidden_size)), requires_grad=True)

    def forward(self, X):
        """
        Activation of this layer.

        Parameters
        ----------
        X : Tensor or array-like
            Input at this timestep

        Returns
        -------
        Tensor
            The final hidden state vector of shape '(batch_size, hidden_size)'

        Notes
        -----
        The forward unrolls the recurrent cell across all timesteps.
        At each timestep t, the hidden state is computed via:
            h_t = tanh(X_t @ W_xh + h_(t-1) @ W_hh)

        This layer implements a Many-toOne configuration by returning only fhe final hidden_stat
        h_t after processing the whole sequence.

        """
        X_t = X if isinstance(X, Tensor) else Tensor(X)
        batch_size = X_t.shape[0]
        timestep = X_t.shape[1]

        h_states = [Tensor(np.zeros((batch_size, self.hidden_size)))]

        for t in range(timestep):
            x_t = X_t[:, t, :]
            # x_t @ self.W_xh : (batch, input_size) @ (input_size, hidden_size) -> (batch, hidden_size)
            # h_prev @ self.W_hh : (batch, hidden_size) @ (hidden_size, hidden_size) -> (batch, hidden_size)
            h_next = (x_t @ self.W_xh + h_states[-1] @ self.W_hh + self.b_h).tanh()
            h_states.append(h_next)

        return h_states[-1]

    def get_params(self):
        """Return a list of trainable parameters."""
        return [self.W_xh, self.W_hh, self.b_h]

    def save_state(self):
        """Return a copy of the underlying weights and biases."""
        return {
            "W_xh": self.W_xh.data.copy(),
            "W_hh": self.W_hh.data.copy(),
            "b_h": self.b_h.data.copy(),
        }

    def load_state(self, state):
        """
        Load parameter data safely into the existing Tensor instances.

        Parameters
        ----------
        state : dict
            A dictionary containing the state arrays
        """
        self.W_xh = Tensor(state["W_xh"].copy(), requires_grad=True)
        self.W_hh = Tensor(state["W_hh"].copy(), requires_grad=True)
        self.b_h = Tensor(state["b_h"].copy(), requires_grad=True)

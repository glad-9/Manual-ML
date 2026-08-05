from .base import Optimizer
import cupy as cp


class Adagrad(Optimizer):
    """
    AdaGrad (Adaptive Gradient Algorithm) Optimizer.

    Adagrad uses an adaptive, per-parameter learning rate instead of a single, fixed learning
    rate for all parameters.
    Uses a running accumulation of gradient variances per parameters to determine the learning rate.

    """

    def __init__(self, **kwargs):
        """
        Initializes an Adagrad type Optimizer object

        Parameters
        ----------
        **kwargs
            Passed to Optimizer base class. See nn/optim/base.py

        Attributes
        ----------
        sq_grads : python dict
            Running accumulation of squared gradients per parameter (no decay)

        epsilon : float
            Small number to prevent division by zero
            Set to 1e-8
        """
        super().__init__(**kwargs)

        self.sq_grads = {}  # Maintains a running average of squared gradients
        self.epsilon = 1e-8

    def step(self, layers):
        """
        Parameters
        ----------
        layers : List of Layer objects

        Returns
        -------
        None

        Notes
        -----
        Update Rule:
            sq_grads += grad ** 2

            Parameter Update: p = p - (lr * grad) / sqrt(sq_grads + epsilon)

        Gradient clipping at [-5, 5] to prevent instability.

        """
        for layer in layers:
            for p in layer.get_params():
                key = id(p)
                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = cp.clip(p.grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = cp.zeros_like(p.data)

                self.sq_grads[key] += grad**2
                p.data -= (self.lr * grad) / cp.sqrt(self.sq_grads[key] + self.epsilon)
                p.zero_grad()

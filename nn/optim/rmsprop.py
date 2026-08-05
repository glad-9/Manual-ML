from .base import Optimizer
import cupy as cp


class RMSprop(Optimizer):
    """
    RMSprop Optimizer.

    Designed to fix the issues of the Adagrad algorithm.
    It adds a decay rate to the equation which acts as a forgetting factor (beta, usually set to 0.9).
    Scales learning rate based on the gradient variance and adapts to change as a result of
    the decay rate which lets older gradients decay while giving more weight to recent ones.
    Because of the decay rate, the denominator in RMSprop (v_next) does not grow forever,
    solving the problem of radically diminishing or exploding learning rates during training.

    Attributes
    ----------
    dr : float
        Used in calculating the second moment - v_next
    sq_grads : python dict
        Maintains a running average of squared gradients
    """

    def __init__(self, dr=0.9, **kwargs):
        """
        Initializes an RMSprop Optimizer.

        Parameters
        ----------
        dr : float
            Decay Rate: Controls how much weight the optimizer gives to recent gradients compared to older gradients when dynamically scaling the learning rate
            Default is 0.9

        **kwargs
            Passed to Optimizer base class. See nn/optim/base.py

        """
        super().__init__(**kwargs)

        self.dr = dr  # Decay Rate
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
            v = (dr * v_prev) + (1 - dr) * (grad ** 2)
            p = p - (lr * grad) / sqrt(v + epsilon)

        Gradients are clipped to [-5, 5] before the update to prevent instability from large gradients. epsilon prevents division by zero when v is small.
        """
        for layer in layers:
            for p in layer.get_params():
                key = id(p)
                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = cp.clip(p.grad, -5, 5)

                if key not in self.sq_grads:
                    self.sq_grads[key] = cp.zeros_like(p.data)

                v_prev = self.sq_grads[key]

                v_next = (self.dr * v_prev) + (1 - self.dr) * (grad**2)

                self.sq_grads[key] = v_next
                p.data -= (self.lr * grad) / cp.sqrt(v_next + self.epsilon)
                p.zero_grad()

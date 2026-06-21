from .base import Optimizer


class SGD(Optimizer):
    def __init__(self, **kwargs):
        """
        Initialize an SGD type Optimizer object

        Parameters
        ----------
        **kwargs
            Passed to Optimizer base class. See nn/optim/base.py.
        """
        super().__init__(**kwargs)

    def step(self, layers):
        """
        Notes
        -----
        Update Rule: p = p - lr * grad

        If requires_reg is True on a parameter, L2 regularization (weight decay) is applied before the update by adding reg * p.data to the gradient, which penalizes large weights

        """
        for layer in layers:
            for p in layer.get_params():
                if p.requires_reg:
                    p.grad += self.reg * p.data

                p.data -= self.lr * p.grad
                p.zero_grad()

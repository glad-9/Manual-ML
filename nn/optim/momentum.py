from .base import Optimizer
import cupy as cp


class Momentum(Optimizer):
    def __init__(self, mf=0.9, **kwargs):
        """
        Initialize a Momentum type Optimizer object

        Parameters
        ----------
        mf : float
            Sets the momentum factor to a certain value
            Determines how much of the previous gradient step is carried to the next
            Default is 0.9

        **kwargs
            Passed to Optimizer base and its base class. See nn/optim/base.py.

        Attributes
        ----------
        mf : float
            Used in calculating the next moment
        momentum : python dict
            Maintains a running average of all gradients for each parameter
        """

        super().__init__(**kwargs)

        self.mf = mf
        self.momentum = {}

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
            m = mf * m_prev + (1 - mf) * grad
            p = p - lr * m

        Gradients are clipped to [-5, 5] to prevent instability from large gradients. A per-parameter running average is keyed by id(p) and persists across steps
        """
        for layer in layers:
            for p in layer.get_params():
                key = id(p)

                if p.requires_reg:
                    p.grad += self.reg * p.data

                grad = cp.clip(p.grad, -5, 5)

                if key not in self.momentum:
                    self.momentum[key] = cp.zeros_like(p.data)

                m_prev = self.momentum[key]

                m_next = (self.mf * m_prev) + (1 - self.mf) * grad

                self.momentum[key] = m_next
                p.data -= self.lr * m_next
                p.zero_grad()

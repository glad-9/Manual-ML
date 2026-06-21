from .base import Optimizer
import cupy as cp


class Adam(Optimizer):
    def __init__(self, mf=0.9, dr=0.99, **kwargs):
        """
        Initializes an Adam type Optimizer object

        Parameters
        ----------
        mf : float
            Sets the momentum factor to a certain value
            Determines how much of the previous gradient step is carried to the next
            Default is 0.9

        dr : float
            Sets the decay rate to a certain value
            Controls how much weight the optimizer gives to recent gradients compared to older gradients when dynamically scaling the learning rate
            Default is 0.99

        **kwargs
            Passed in Optimizer base class. See nn/optim/base.py

        Attributes
        ----------
        mf : float
            Used in calculating the first moment

        dr : float
            Used in calculating the second moment

        momentum : python dict
            Running average of gradients per parameter

        velocity : python dict
            Running average of squared gradients per parameter

        t : int
            Tracks the number of steps taken by the optimizer

        epsilon : float
            Small number to prevent division by zero
            Set to 1e-8
        """

        super().__init__(**kwargs)

        self.mf = mf  # mf = momentum factor
        self.dr = dr  # dr = decay rate
        self.momentum = {}  # Maintains a running average of gradients
        self.velocity = {}  # Maintains a running average of squared gradients
        self.t = 0  # Step counter
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
            First moment (momentum): m = mf * m_prev + (1 - mf) * grad
            Second moment (variance): v = dr * v_prev + (1 - dr) * (grad ** 2)
            Bias correction:
                m_hat = m / (1 - mf ** t)
                v_hat = v / (1 - dr ** t)

            Parameter update: p = p - (lr * m_hat) / (sqrt(v_hat) + epsilon)

        m and v start at zero, which biases early estimates toward zero - bias correction counteracts this, with the effect shrinking as t grows.
        Gradients are clipped to [-5, 5] to prevent instability.
        Epsilon prevents division by zero

        """
        self.t += 1
        for layer in layers:
            for p in layer.get_params():
                if p.requires_reg:
                    p.grad += self.reg * p.data

                key = id(p)
                grad = cp.clip(p.grad, -5, 5)

                # First moment estimate

                if key not in self.momentum:
                    self.momentum[key] = cp.zeros_like(p.data)

                m_prev = self.momentum[key]

                m_raw = (self.mf * m_prev) + (1 - self.mf) * grad

                self.momentum[key] = m_raw

                # Second moment estimate
                if key not in self.velocity:
                    self.velocity[key] = cp.zeros_like(p.data)

                v_prev = self.velocity[key]

                v_raw = (self.dr * v_prev) + (1 - self.dr) * (grad**2)

                self.velocity[key] = v_raw

                # Bias correction
                m_corrected = m_raw / (1 - self.mf**self.t)
                v_corrected = v_raw / (1 - self.dr**self.t)

                p.data -= (self.lr * m_corrected) / (
                    cp.sqrt(v_corrected) + self.epsilon
                )
                p.zero_grad()

from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Abstract base class for optimizers that update layers parameters using gradients computed during backpropagation.

    Subclasses implement step() to define a specific update rule
    (e.g. SGD, Momentum, Adam).

    Contract
    --------
    Each layer passed to step() must implement get_params(), returning a list
    of Tensor objects representing its learnable parameters

    Each parameter Tensor is expected to have:
        - .grad : ndarray, populated via backward() before step() is called
        - .requires_reg : bool, whether L2 weight decay is applied to this param
        - .zero_grad() : resets .grad to None after the update is applied

    An implementation of an optimizer must call zero_grad() at the end of a parameter update because it is not done by the network.
    """

    def __init__(self, lr: float = 0.01, reg: float = 0.0):
        """
        An Optimizer Base class for other types to inherit from

        Parameters
        ----------
        lr : float
            Sets the learning rate to a certain value
            Determines the rate at which the network learns
            Default is 0.01
        reg: float
            Sets the weight decay to a certain value
            Determines the scale of the weight penalty - increase when overfitting
            Default is 0.0

        Attributes
        ----------
        lr : float

        """
        self.lr = lr
        self.reg = reg

    @abstractmethod
    def step(self, layers):
        """
        Defines one step of backpropagation after computing the cost
        Subclasses implement the specific update rule

        Parameters
        ----------
        layers : List of Layer objects

        Returns
        -------
        None
        """
        pass


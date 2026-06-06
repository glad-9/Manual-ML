import numpy as np


class Tensor:
    def __init__(self, data, requires_grad=False, requires_reg=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self.requires_reg = requires_reg

        self._backward = lambda: None
        self._prev = set()

        self.op = "leaf"

    def backward(self):
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if not node.requires_grad:
                continue

            node._backward()

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return Tensor(self.data[idx])

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return

        grad = self._unbroadcast(grad, self.data.shape)

        if self.grad is None:
            self.grad = np.zeros_like(self.data).astype("float32")

        self.grad += grad
        assert self.grad.shape == self.data.shape

    def _create_results(self, data, *parents):
        requires_grad = any(p.requires_grad for p in parents)

        out = Tensor(data, requires_grad=requires_grad)
        out._prev = set(parents)

        return out

    def _unbroadcast(self, grad, shape):
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def zero_grad(self):
        self.grad = None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data + other.data
        out = self._create_results(op, self, other)
        out.op = "add"

        def _backward():
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data * other.data
        out = self._create_results(op, self, other)

        out.op = "mul"

        def _backward():
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        op = self.data @ other.data
        out = self._create_results(op, self, other)

        out.op = "matmul"

        def _backward():
            self._accumulate_grad(out.grad @ other.data.T)
            other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data**other.data
        out = self._create_results(op, self, other)

        out.op = "pow"

        def _backward():
            self._accumulate_grad(
                other.data * (self.data ** (other.data - 1)) * out.grad
            )
            other._accumulate_grad(
                np.log(self.data) * (self.data**other.data) * out.grad
            )

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return self * (other**-1.0)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other * (self**-1.0)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other + (-self)

    def __neg__(self):
        return self * -1.0

    def log(self):
        op = np.log(self.data)
        out = self._create_results(op, self)
        out.op = "log"

        def _backward():
            self._accumulate_grad(np.reciprocal(self.data) * out.grad)

        out._backward = _backward
        return out

    def exp(self):
        op = np.exp(self.data)
        out = self._create_results(op, self)
        out.op = "exp"

        def _backward():
            self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None):
        op = (
            np.sum(self.data)
            if axis is None
            else np.sum(self.data, axis=axis, keepdims=True)
        )
        out = self._create_results(op, self)
        out.op = "sum"

        def _backward():
            self._accumulate_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self):
        n = self.data.size
        return self.sum() / n

    def var(self):
        mean = self.mean()
        return ((self - mean) ** 2).mean()

    def relu(self):
        op = np.maximum(0, self.data)
        out = self._create_results(op, self)

        def _backward():
            self._accumulate_grad(out.grad * (out.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self):
        x = self.data
        op = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
        out = self._create_results(op, self)

        def _backward():
            self._accumulate_grad(out.grad * (op * (1 - op)))

        out._backward = _backward
        return out

    def tanh(self):
        e1 = self.exp()
        e2 = (-self).exp()

        return (e1 - e2) / (e1 + e2)

    def softmax(self):
        shifted = self - self.data.max(axis=1, keepdims=True)
        exp = shifted.exp()
        return exp / exp.sum(axis=1)

    def bce(self, y):
        self.data = np.clip(self.data, 1e-7, 1 - 1e-7)
        return -(y * self.log() + (Tensor(1.0) - y) * (Tensor(1.0) - self).log()).mean()

    def mse(self, y):
        return (Tensor(0.5) * ((self - y) ** 2)).mean()

    def ce(self, y):
        return -(y * self.log()).sum(axis=1).mean()

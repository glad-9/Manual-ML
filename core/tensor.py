import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set()

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
            node._backward()

    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return Tensor(self.data[idx])

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        self.grad += grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data + other.data)
        out._prev = {self, other}

        def _backward():
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data * other.data)
        out._prev = {self, other}

        def _backward():
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        out._prev = {self, other}

        def _backward():
            self._accumulate_grad(out.grad @ other.data.T)
            other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        out = Tensor(self.data ** other.data)
        out._prev = {self, other}

        def _backward():
            self._accumulate_grad(other.data * (self.data ** (other.data-1)) * out.grad)
            other._accumulate_grad(np.log(self.data) * (self.data ** other.data) * out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return self * (other ** -1)

    def __sub__(self, other):
        return self + (-other)


    def sum(self):
        out = Tensor(np.sum(self.data, axis=0, keepdims=True))
        out._prev = {self}

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def mean(self):
        m = self.data.size
        out = Tensor(np.mean(self.data, axis=0, keepdims=True))
        out._prev = {self}

        def _backward():
            self.grad += (1.0 / m) * np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data))
        out._prev = {self}

        def _backward():
            self._accumulate_grad(np.reciprocal(self.data) * out.grad)

        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data))
        out._prev = {self}

        def _backward():
            self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data))
        out._prev = {self}

        def _backward():
            self._accumulate_grad(out.grad * (out.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self):
        return Tensor(1.0) / (Tensor(1.0) + (-self).exp())

    def tanh(self):
        return (self.exp() - (-self).exp()) / (self.exp() + (-self).exp())

    def bce(self, y):
        return -(y * self.log() + (Tensor(1.0) - y) * (Tensor(1.0) - self).log()).mean()

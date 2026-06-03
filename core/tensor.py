import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad

        self._backward = lambda: None
        self._prev = set()

        self.op = 'leaf'

    def backward(self): 
        topo = [] 
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)
                # print(
                #     f"id={id(node)} "
                #     f"op={node.op} "
                #     f"shape={node.data.shape if hasattr(node.data, 'shape') else node.data}"
                # )
        
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if not node.requires_grad:
                continue

            # print(f"calling backward on id: {id(node)}, shape: {node.data.shape if hasattr(node.data, 'shape') else node.data}, requires_grad: {node.requires_grad}")
            node._backward()

    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        return Tensor(self.data[idx])

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return

        grad = self._unbroadcast(grad, self.data.shape)
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data).astype('float64')

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

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data + other.data
        out = self._create_results(op, self, other)
        out.op = 'add'

        def _backward():
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data * other.data
        out = self._create_results(op, self, other)

        out.op = 'mul'

        def _backward():
            # print(f"mul: out id={id(out)}, out.grad={out.grad.shape if hasattr(out.grad, 'shape') else out.grad}")
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)

        out._backward = _backward
        return out


    def __matmul__(self, other):
        op = self.data @ other.data
        out = self._create_results(op, self, other)

        out.op = 'matmul'

        def _backward():
            self._accumulate_grad(out.grad @ other.data.T)
            other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        op = self.data ** other.data
        out = self._create_results(op, self, other)

        out.op = 'pow'

        def _backward():
            self._accumulate_grad(other.data * (self.data ** (other.data-1)) * out.grad)
            other._accumulate_grad(np.log(self.data) * (self.data ** other.data) * out.grad)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return self * (other ** -1)

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other * (self ** -1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other))
        return other + (-self)

    def __neg__(self):
        return self * -1


    def sum(self):
        op = np.sum(self.data, axis=0, keepdims=True)
        out = self._create_results(op, self)
        out.op = 'sum'

        def _backward():
            self._accumulate_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self):
        m = self.data.size
        op = np.mean(self.data, axis=0, keepdims=True)
        out = self._create_results(op, self)
        out.op = 'mean'

        def _backward():
            # print(f"mean: out id={id(out)}, out.grad={out.grad}")
            self._accumulate_grad((1.0 / m) * np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def log(self):
        op = np.log(self.data)
        out = self._create_results(op, self)
        out.op = 'log'

        def _backward():
            self._accumulate_grad(np.reciprocal(self.data) * out.grad)

        out._backward = _backward
        return out

    def exp(self):
        op = np.exp(self.data)
        out = self._create_results(op, self)
        out.op = 'exp'

        def _backward():
            self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def relu(self):
        op = np.maximum(0, self.data)
        out = self._create_results(op, self)

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

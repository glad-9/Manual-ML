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
        return Tensor(
            self.data[idx],
            requires_grad=self.requires_grad,
            requires_reg=self.requires_reg,
        )

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return

        grad = np.clip(grad, -1, 1)
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
            self._accumulate_grad(out.grad @ other.data.swapaxes(-1, -2))
            other._accumulate_grad(self.data.swapaxes(-1, -2) @ out.grad)

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

    def reshape(self, *shape):
        op = self.data.reshape(shape)
        out = self._create_results(op, self)
        out.op = "reshape"

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        op = self.data.transpose(axes)
        out = self._create_results(op, self)
        out.op = "transpose"

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad.transpose(np.argsort(axes)))

        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        op = np.clip(self.data, min_val, max_val)
        out = self._create_results(op, self)

        def _backward():
            mask = (self.data >= min_val) & (self.data <= max_val)
            self._accumulate_grad(out.grad * mask)

        out._backward = _backward

        return out

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

    def sum(self, axis=None, keepdims=True):
        op = (
            np.sum(self.data)
            if axis is None
            else np.sum(self.data, axis=axis, keepdims=keepdims)
        )
        out = self._create_results(op, self)
        out.op = "sum"

        def _backward():
            self._accumulate_grad(np.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=True):
        n = self.data.size
        return self.sum(axis, keepdims) / n

    def var(self, axis=None, keepdims=True):
        mean = self.mean(axis, keepdims)
        return ((self - mean) ** 2).mean()

    def std(self, axis=None, keepdims=True, epsilon=1e-8):
        var = self.var(axis=axis, keepdims=keepdims)
        return (var + epsilon) ** 0.5

    def pad2d(self, p):
        n, c, h, w = self.data.shape
        padded = np.zeros((n, c, h + (2 * p), w + (2 * p)), dtype=self.data.dtype)
        padded[:, :, p : p + h, p : p + w] = self.data
        out = self._create_results(padded, self)

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad[:, :, p : h + p, p : w + p])

        out._backward = _backward
        return out

    def pool2d(self, pool_size=2, stride=None):
        n, c, h, w = self.data.shape

        if pool_size is None:
            pool_size = 2

        if stride is None:
            stride = pool_size

        h_out = ((h - pool_size) // stride) + 1
        w_out = ((w - pool_size) // stride) + 1

        out_data = np.zeros((n, c, h_out, w_out))
        mask = np.zeros_like(self.data)  # Remember where maxes are

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride

                patch = self.data[
                    :, :, h_start : h_start + pool_size, w_start : w_start + pool_size
                ]

                # Get max value per patch whilst keeping dimensionality
                max_vals = np.max(patch, axis=(2, 3), keepdims=True)  # (n, c, 1, 1)

                out_data[:, :, i, j] = max_vals[:, :, 0, 0]

                # Create position mask by checking what value in the patch is equivalent to the max
                mask[
                    :, :, h_start : h_start + pool_size, w_start : w_start + pool_size
                ] = patch == max_vals

        out = self._create_results(out_data, self)

        def _backward():
            assert out.grad is not None
            grad = np.zeros_like(self.data)
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * stride
                    w_start = j * stride

                    grad[
                        :,
                        :,
                        h_start : h_start + pool_size,
                        w_start : w_start + pool_size,
                    ] += (
                        mask[
                            :,
                            :,
                            h_start : h_start + pool_size,
                            w_start : w_start + pool_size,
                        ]
                        * out.grad[:, :, i : i + 1, j : j + 1]
                    )
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def im2col(self, kH, kW, stride, h_out, w_out):
        n, c, h, w = self.data.shape
        col = np.zeros((n, h_out * w_out, kH * kW * c))

        for i in range(h_out):
            for j in range(w_out):
                h_start = i * stride
                w_start = j * stride

                patch = self.data[:, :, h_start : h_start + kH, w_start : w_start + kW]
                col[:, i * w_out + j, :] = patch.reshape(n, -1)

        out = self._create_results(col, self)
        out.op = "im2col"

        def _backward():
            assert out.grad is not None
            grad = np.zeros_like(self.data)
            for i in range(h_out):
                for j in range(w_out):
                    h_start = i * stride
                    w_start = j * stride
                    patch_grad = out.grad[:, i * w_out + j, :].reshape(n, c, kH, kW)
                    grad[:, :, h_start : h_start + kH, w_start : w_start + kW] += (
                        patch_grad
                    )
            self._accumulate_grad(grad)

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
        clipped = self.clip(1e-7, 1 - 1e-7)
        return -(
            y * clipped.log() + (Tensor(1.0) - y) * (Tensor(1.0) - clipped).log()
        ).mean()

    def mse(self, y):
        return (Tensor(0.5) * ((self - y) ** 2)).mean()

    def ce(self, y):
        clipped = self.clip(1e-7, 1.0)
        return -(y * clipped.log()).sum(axis=1).mean()

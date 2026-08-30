from core.backend import xp
from core.backend import scatter_add
from core.backend import as_strided

assert xp is not None


class Tensor:
    """
    A multi-dimensional array wrapper with automatic differentiation capabilities through a computation graph.

    Attributes
    ----------
    data: xp.ndarray
        The multi-dimensional array containing the tensor values
    grad: ndarray or None
        Gradient of loss with respect to this tensor. None until backward() is called
    requires_grad : bool
        True if gradients need to be calculated during backpropagation
    requires_reg : bool
        True if the tensor is flagged for weight decay optimization
    op: str
        String label for the operation that produced this tensor.
    _prev : set
        The parent tensors in the computation graph that this tensor was derived from
    _backward : function
        Internal closure that handles gradient accumulation for this specific node/Tensor
    """

    def __init__(self, data, requires_grad=False, requires_reg=False):
        """
        Initialize a Tensor object.

        Parameters
        ----------
        data : array-like
            The underlying data for the tensor. Can be a list, NumPy array, or CuPy array.
            Will be converted to CuPy array with float32 dtype.

        requires_grad : bool, optional
            Whether gradients should be computed for this tensor during backward pass.
            Default is False

        requires_reg : bool, optional
            Flag used by an optimizer to determine whether a Tensor experiences weight decay
            Default is False

        """
        self.data = xp.asarray(data, dtype=xp.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self.requires_reg = requires_reg

        self._backward = lambda: None
        self._prev = set()

        self.op = "leaf"

    def backward(self):
        """
        Start the backpropagation chain

        Computes the gradients for all Tensors in the computation graph where 'requires_grad' is True. This method performs a topological sort to ensure parent nodes accumulate upstream gradients before executing their own backward closures.

        Returns
        -------
        None

        Notes
        -----
        This method must be called on the scalar output of a loss function.
        It initializes the root gradient as an array of ones ('xp.ones_like') then traverses the graph in reverse topological order, calling the node/Tensor's internal '_backward()' closure to propagate derivatives.
        """
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)

                topo.append(node)

        build_topo(self)
        self.grad = xp.ones_like(self.data)
        for node in reversed(topo):
            if not node.requires_grad:
                continue

            node._backward()

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, idx):
        """
        Slice or index the Tensor

        Supports standard Python slicing and advanced indexing rules while maintaining structural positioning within the computation graph.

        Parameters
        ----------
        idx : int, slice, or tuple
            The index or slice used to extract data.

        Returns
        -------
        Tensors
            A new view or copy of the sliced sub-tensor, tracked in the graph.

        Notes
        -----
        Forward pass: Extracts a subset or view of the underlying array using 'self.data[idx]'
        Backward Pass:
        The gradient of a sliced tensor is sparse. The incoming upstream gradient must be accumulated back into an array of the original shape at the indices specified by idx. All other indices receive zero gradient.
        """
        op = self.data[idx]
        out = self._create_results(op, self)
        out.op = "getitem"

        def _backward():
            # Create a zero array matching the parent's shape
            full_grad = xp.zeros_like(self.data)

            # Route the sliced upstream gradient into its original index position
            full_grad[idx] = out.grad
            self._accumulate_grad(full_grad)

        out._backward = _backward
        return out

    def _accumulate_grad(self, grad):
        """
        Helper method to accumulate gradients

        Returns
        -------
        None

        Notes
        -----
        Accumulates gradients for Tensors where 'requires_grad' is set to True.
        Gradients initialized to None are converted to zero arrays.
        """
        if not self.requires_grad:
            return

        grad = self._unbroadcast(grad)

        if self.grad is None:
            self.grad = xp.zeros_like(self.data).astype("float32")

        self.grad += grad
        assert self.grad.shape == self.data.shape

    def _create_results(self, data, *parents):
        """
        Creates a Tensor with requires_grad inherited from its parents and populated _prev with parent Tensors

        Parameters
        ----------
        data : xp.ndarray
            Computed data of the operation/method

        *parents : tuple of Tensor
            Tuple of parent Tensors that were used to create the resulting Tensor

        Returns
        -------
        Tensor
            Output Tensor with necessary requires_grad and _prev parents set populated
        """
        requires_grad = any(p.requires_grad for p in parents)

        out = Tensor(data, requires_grad=requires_grad)
        out._prev = set(parents)

        return out

    def _unbroadcast(self, grad):
        """
        Reduces a broadcasted gradient back to the original tensor shape.

        Parameters
        ----------
        grad : xp.ndarray
            The gradient in its (potentially) expanded/broadcasted shape.

        Returns
        -------
        xp.ndarray
            The gradient reshaped and summed to match shape.

        Notes
        -----
        When an operation (like addition) involves broadcasting, the resulting
        gradient in out.grad has the larger, broadcasted shape. This method
        reverts that expansion by summing gradients across all axes where
        broadcasting occurred.

        """
        shape = self.data.shape
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    def zero_grad(self):
        """
        Reset gradients of this Tensor

        Returns
        -------
        None

        Notes
        -----
        After one step of gradient descent, gradients must be reset to None to prevent accumulating on gradients from previous steps
        """
        self.grad = None

    def __add__(self, other):
        """
        Element-wise addition of two tensors.

        Overrides '+' and performs element-wise addition with broadcasting support

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to add to this tensor.

        Returns
        -------
        Tensor
            A new tensor containing the element-wise sum.

        Notes
        -----
        Forward Pass: Computes self.data + other.data
        Backward Pass:
        The coefficient of both operands is 1
        Changing either does not explicitly affect the other.

        - Gradient w.r.t self: receives out.grad unchanged
        - Gradient w.r.t other: receives out.grad unchanged

        Broadcasting: Supports NumPy broadcasting rules. For gradient reduction following broadcasting, see _unbroadcast

        Examples
        --------
        >>> a = Tensor([1,2,3])
        >>> b = Tensor([4,5,6])
        >>> c = a + b
        >>> print(f"Data: {c.data}")
        Data: [5. 7. 9.]
        >>> print(f"Gradients:\na:{a.grad}\nb:{b.grad}\nc:{c.grad}")
        Gradients:
        a:[1. 1. 1.]
        b:[1. 1. 1.]
        c:[1. 1. 1.]
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        op = self.data + other.data
        out = self._create_results(op, self, other)
        out.op = "add"

        def _backward():
            self._accumulate_grad(out.grad)
            other._accumulate_grad(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Element-wise multiplication of two tensors.

        Overrides '*' and performs element-wise multiplication with broadcasting support

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to multiply with this tensor.

        Returns
        -------
        Tensor
            A new tensor containing the element-wise product.

        Notes
        -----
        Forward Pass: Computes self.data * other.data
        Backward Pass:
        By the product rule, d/dx(x * y) = y and d/dy (x * y) = x,
        so each input receives the upstream gradient scaled by the other operand

        - Gradient w.r.t self: other.data * out.grad
        - Gradient w.r.t other: self.data * out.grad

        Broadcasting: Supports NumPy broadcasting rules. For gradient reduction following broadcasting, see _unbroadcast

        Examples
        --------
        >>> a = Tensor([1,2,3])
        >>> b = Tensor([4,5,6])
        >>> c = a * b
        >>> c.backward()
        >>> print(f"Data: {c.data}")
        Data: [4. 10. 18.]
        >>> print(f"Gradients:\na:{a.grad}\nb:{b.grad}\nc:{c.grad}")
        Gradients:
        a:[4. 5. 6.]
        b:[1. 2. 3.]
        c:[1. 1. 1.]
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        op = self.data * other.data
        out = self._create_results(op, self, other)

        out.op = "mul"

        def _backward():
            self._accumulate_grad(other.data * out.grad)
            other._accumulate_grad(self.data * out.grad)

        out._backward = _backward
        return out

    def __matmul__(self, other):
        """
        Matrix multiplication of two tensors.

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to matrix multiply with this tensor.

        Returns
        -------
        Tensor
            A new tensor containing the matrix product.

        Notes
        -----
        Forward Pass: self.data @ other.data
        Backward Pass:
        For Z = X @ W, dL/dX  = dL/dZ @ W.T and dL/dW = X.T @ dL/dZ
        'swapaxes(-1, -2)' is used instead of .T to handle higher dimensional arrays with batch dimensions.
        - Gradient w.r.t self: out.grad @ other.data.swapaxes(-1, -2)
        - Gradient w.r.t other: self.data.swapaxes(-1, -2) @ out.grad

        Examples
        --------
        >>> a = Tensor(np.array([[1.0], [2.0]]), requires_grad=True)
        >>> b = Tensor(np.array([[3.0, 4.0],]), requires_grad=True)
        >>> c = a @ b
        >>> c.backward()
        >>> print("Data:", c.data)
        Data: [[3. 4.] [6. 8.]]
        >>> print(f"Gradients:\na:{a.grad}\nb:{b.grad}\nc:{c.grad}")
        Gradients:
        a:[[7.] [7.]]
        b:[[3. 3.]]
        c:[[1. 1.] [1. 1.]]
        """
        op = self.data @ other.data
        out = self._create_results(op, self, other)

        out.op = "matmul"

        def _backward():
            self._accumulate_grad(out.grad @ other.data.swapaxes(-1, -2))
            other._accumulate_grad(self.data.swapaxes(-1, -2) @ out.grad)

        out._backward = _backward
        return out

    def __pow__(self, other):
        """
        Element-wise exponentiation of this tensor by another tensor or scalar.

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to raise this tensor to the power of.

        Returns
        -------
        Tensor
            A new tensor containing the element-wise power operation.

        Notes
        -----
        Forward Pass: self.data ** other.data
        Backward Pass:
            By the calculus power rule,
            d/dx (x ** y) = y * (x ** (y - 1))
            d/dy (x ** y) = ln(x) * (x ** y)

        - Gradient w.r.t self: out.grad * (other.data) * (self.data ** (other.data - 1))
        - Gradient w.r.t other: out.grad * xp.log(self.data) * (self.data ** other.data)

        Broadcasting: Supports NumPy broadcasting rules. For gradient reduction following broadcasting, see _unbroadcast

        Examples
        --------
        >>> a = Tensor(np.array([[1.0], [2.0]]), requires_grad=True)
        >>> b = Tensor(np.array([[3.0, 4.0],]), requires_grad=True)
        >>> c = a @ b
        >>> c.backward()
        >>> print("Data:", c.data)
        Data: [[ 1.] [ 4.] [ 9.] [16.]]
        >>> print(f"Gradients:\na:{a.grad}\nb:{b.grad}\nc:{c.grad}")
        Gradients:
        a:[[2.] [4.] [6.] [8.]]
        b:None
        c:[[1.] [1.] [1.] [1.]]
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        op = self.data**other.data
        out = self._create_results(op, self, other)

        out.op = "pow"

        def _backward():
            self._accumulate_grad(
                other.data * (self.data ** (other.data - 1)) * out.grad
            )
            other._accumulate_grad(xp.log(self.data) * op * out.grad)

        out._backward = _backward
        return out

    def __neg__(self):
        """
        Unary negation.

        Computes the element-wise additive inverse of the tensor.

        Returns
        -------
        Tensor
            A new Tensor containing the negated values.

        Notes
        -----
        Implemented via element-wise multiplication by -1.0, introducing a multiplication node into the computation graph.

        See Also
        --------
        mul : The primary implementation of element-wise multiplication
        """
        return self * -1.0

    def __sub__(self, other):
        """
        Element-wise subtraction.

        Computes self - other by performing element-wise addition with the negated version of other.

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to subtract from this tensor

        Returns
        -------
        Tensor
            A new tensor containing the result of the subtraction.

        Notes
        -----
        In the computation graph, this operation introduces both a unary negation node and an adddition node.

        See Also
        --------
        add : The primary implementation of element-wise addition
        mul : The primary implementation of element-wise multiplication
        """
        return self + (-other)

    def __truediv__(self, other):
        """
        Element-wise division.

        Computes self / other by performing element-wise multiplication with the reciprocal of the other.

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to divide this tensor by

        Returns
        -------
        Tensor
            A new tensor containing the result of the element-wise division

        Notes
        -----
        In the computation graph, this operation introduces a multiplication and a power node.

        See Also
        --------
        mul : The primary implementation of element-wise multiplication
        pow : The primary implementation of element-wise power
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        return self * (other**-1.0)

    def __radd__(self, other):
        """
        Reflected element-wise addition.

        Triggered when a non-Tensor object is on the left side of the '+' operator.
        The operation logic and gradient are identical to add

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to add with this tensor.

        Returns
        -------
        Tensor
            A new tensor containing the element-wise addition

        See Also
        --------
        add : The primary implementation of element-wise addition
        """
        return self + other

    def __rmul__(self, other):
        """
        Reflected element-wise multiplication.

        Triggered when a non-Tensor object is on the left side of the '*' operator.
        The operation logic and gradient are identical to mul

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to multiply with this tensor.

        Returns
        -------
        Tensor
            A new tensor containing the element-wise product.

        See Also
        --------
        mul : The primary implementation of element-wise multiplication
        """
        return self * other

    def __rsub__(self, other):
        """
        Reflected element-wise subtraction.

        Triggered when a non-Tensor object is on the left side of the '-' operator.
        The operation logic and gradient are identical to sub

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to subtract from this tensor

        Returns
        -------
        Tensor
            A new tensor containing the result of the subtraction.

        See Also
        --------
        sub : The primary implementation of element-wise subtraction
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        return other + (-self)

    def __rtruediv__(self, other):
        """
        Reflected element-wise division.

        Triggered when a non-Tensor object is on the left side of the '/' operator.
        The operation logic and gradient are identical to sub

        Parameters
        ----------
        other : Tensor or array-like
            The tensor or scalar to divide this tensor by

        Returns
        -------
        Tensor
            A new tensor containing the result of the division.

        See Also
        --------
        truediv : The primary implementation of element-wise division.
        """
        other = other if isinstance(other, Tensor) else Tensor(xp.array(other))
        return other * (self**-1.0)

    def reshape(self, *shape):
        """
        Reshape the tensor to specified shape

        Parameters
        ----------
        *shape : Tuple of int
            The new shape for the tensor. Can be passed as separate arguments or as a tuple.

        Returns
        -------
        Tensor
            A new tensor with the specified shape.

        Notes
        -----
        Forward Pass: self.data.reshape(shape)
        Backward Pass:
        Reshapes out.grad back into the original shape of self.data
        """
        op = self.data.reshape(shape)
        out = self._create_results(op, self)
        out.op = "reshape"

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad.reshape(self.data.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes):
        """
        Transpose the tensor dimensions with the given axes

        Parameters
        ----------
        *axes : Tuple of int
            The new order of dimensions. If not specified, reverses the dimensions.
        Returns
        -------
        Tensor
            A new tensor with transposed dimensions.

        Notes
        -----
        Forward Pass: Rearranges axes using self.data.transpose(axes)
        Backward Pass:
        The backward pass must undo the forward shuffling. This requires calculating the inverse permutation of the forward 'axes' vector.
        Using 'xp.argsort' on the forward axes generates the exact tracking map to restore the upstream gradient back to the tensor's original shape
        """
        op = self.data.transpose(axes)
        out = self._create_results(op, self)
        out.op = "transpose"

        def _backward():
            assert out.grad is not None
            # Convert CuPy argsort array to a Python tuple before transposing - GPU-bound array -> CPU integer args
            axes_order = tuple(xp.argsort(xp.array(axes)).tolist())
            self._accumulate_grad(out.grad.transpose(axes_order))

        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        """
        Clips self.data within a specified range

        Parameters
        ----------
        min_val : float
            Minimum value for self.data elements

        max_val : float
            Maximum value for self.data elements

        Returns
        -------
        Tensor
            Tensor with elements clipped into the range (min_val, max_val)

        Notes
        -----
        Forward Pass (using NumPy/CuPy's clip): xp.clip(self.data, min_val, max_val)
        Backward Pass:
        Creates a mask array where elements in self.data that are between min_val and max_val are backpropagated and the rest are zero.
        """
        op = xp.clip(self.data, min_val, max_val)
        out = self._create_results(op, self)

        def _backward():
            mask = (self.data >= min_val) & (self.data <= max_val)
            self._accumulate_grad(out.grad * mask)

        out._backward = _backward

        return out

    def log(self):
        """
        Uses the natural log (ln) on this Tensor

        Returns
        -------
        Tensor
            Tensor with natural log applied element-wise

        Notes
        -----
        Forward Pass: xp.log(self.data)
        Backward Pass:
        Using the derivative of the natural log we have,
        d/dx (ln(x)) = 1 / x

        - Gradient w.r.t self: (1 / self) * out.grad

        Broadcasting: Supports NumPy broadcasting rules. For gradient reduction following broadcasting, see _unbroadcast
        """
        op = xp.log(self.data)
        out = self._create_results(op, self)
        out.op = "log"

        def _backward():
            self._accumulate_grad(xp.reciprocal(self.data) * out.grad)

        out._backward = _backward
        return out

    def exp(self):
        """
        Exponentiates this Tensor

        Returns
        -------
        Tensor
            Element-wise exponentiated Tensor

        Notes
        -----
        Forward Pass: xp.exp(self.data)
        Backward Pass:
        Using the derivative of e ** x,
        d/dx (e ** x) = x * (e ** x)

        - Gradient w.r.t self: (xp.exp(self.data)) * out.grad

        Broadcasting: Supports NumPy broadcasting rules. For gradient reduction following broadcasting, see _unbroadcast
        """
        op = xp.exp(self.data)
        out = self._create_results(op, self)
        out.op = "exp"

        def _backward():
            self._accumulate_grad(out.data * out.grad)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=True):
        """
        Calculates the sum for this Tensor along the specified axes

        Parameters
        ----------
        axis : Tuple of int or None
            Axis or axes along which to sum. Default is None (all elements).
        keepdims : bool
            Whether to keep original dimensions after the operation

        Return
        ------
        Tensor
            A new tensor summed across the given axes

        Notes
        -----
        Forward Pass: Sums across axes with xp.sum(self.data, axis=axis, keepdims=keepdims)
        Backward Pass:
            Every element of self contributed equally (with coefficient 1) to the sum
            so the upstream gradient is broadcast back unchanged to every position that was
            summed over.

            - Gradient w.r.t self: xp.ones_like(self.data) * out.grad

        Broadcasting: ---

        """
        op = (
            xp.sum(self.data)
            if axis is None
            else xp.sum(self.data, axis=axis, keepdims=keepdims)
        )
        out = self._create_results(op, self)
        out.op = "sum"

        def _backward():
            self._accumulate_grad(xp.ones_like(self.data) * out.grad)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=True):
        """
        Computes the mean of tensor elements along the specified axis

        Parameters
        ----------
        axis : int, tuple of int, or None, optional
            Axis or axes along which to average.
            Default is None (all elements).
        keepdims : bool, optional
           Whether to retain reduced dimensions as size-1 axes.
           Default is True

        Returns
        -------
        Tensor
            A new tensor containing the mean.

        Notes
        -----
        Composed entirely from sum() and division by self.data.size - no backward needed

        See also
        --------
        sum : Primary implementation of the sum operation
        """
        n = self.data.size
        return self.sum(axis, keepdims) / n

    def var(self, axis=None, keepdims=True):
        """
        Computes the variance of tensor elements along the specified axis

        Parameters
        ----------
        axis : int, tuple of int, or None, optional
            Axis or axes along which to calculate the variance.
            Default is None (all elements).
        keepdims : bool, optional
           Whether to retain reduced dimensions as size-1 axes.
           Default is True

        Returns
        -------
        Tensor
            A new tensor with the calculated variance

        Notes
        -----
        Composed from mean() and pow() - no backward needed

        See Also
        --------
        mean : Primary implementation of mean
        """
        mean = self.mean(axis, keepdims)
        return ((self - mean) ** 2).mean()

    def std(self, axis=None, keepdims=True, epsilon=1e-8):
        """
        Computes the standard deviation of tensor elements along the specified axis.

        Parameters
        ----------
        axis : int, tuple of int, or None, optional
            Axis or axes along which to calculate the variance.
            Default is None (all elements).
        keepdims : bool, optional
           Whether to retain reduced dimensions as size-1 axes.
           Default is True.
        epsilon : float
            Extremely small number to prevent division by zero.
            Default is 1e-8.

        Returns
        -------
        Tensor
            A new tensor with the calculated standard deviation

        Notes
        -----
        Composed from var() and pow() - no backward needed

        See Also
        --------
        var : Primary implementation of var
        """
        var = self.var(axis=axis, keepdims=keepdims)
        return (var + epsilon) ** 0.5

    def pad2d(self, p):
        """
        Adds specified padding to this tensor's data

        Parameters
        ----------
        p : int
            Number of pixels to pad by

        Returns
        -------
        Tensor
            A new tensor with the padded shape.

        Notes
        -----
        Forward Pass:
            Increments the dimensions of H and W by 2p and places self.data into the slice [p : p + h, p : p + w], leaving the surrounding border as zeros.

        Backward Pass:
            Slices out.grad to access only self.data, ignoring the padding.
        """
        n, c, h, w = self.data.shape
        padded = xp.zeros((n, c, h + (2 * p), w + (2 * p)), dtype=self.data.dtype)
        padded[:, :, p : p + h, p : p + w] = self.data
        out = self._create_results(padded, self)

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad[:, :, p : h + p, p : w + p])

        out._backward = _backward
        return out

    def im2col(self, kH, kW, stride, h_out, w_out):
        """
        Unfolds the batched image matrix into a 3D tensor.

        Parameters
        ----------
        kH : int
            Kernel height.
        kW : int
            Kernel width.
        stride : int
            Kernel's stride
        h_out : int
            Height of the convolved output matrix
        w_out : int
            Width of the convolved output matrix

        Returns
        -------
        Tensor
            A 3D tensor where each row represents the values of an individual patch (batch_size is the depth)

        Notes
        -----
        Forward Pass:
            Constructs a 6D view of the input using as_strided(), shaped (n, h_out, w_out, c, kh, kw). The view does not copy data - it reinterprets the same underlying buffer by reading it with different strides, so overlapping patches share memory rather than duplicating it. The view is then reshaped to (n, h_out * w_out, patch_values); this reshape forces a real copy, since the strided view is not contiguous.

        Backward Pass:
            Patches overlap whenever stride < kernel size, so a single input pixel can contribute to multiple output patches. The backward pass must therefore sum gradient contributions from every patch a pixel participated in.
            Reverses the forward reshape to recover (n, h_out, w_out, c, kH, kW), then transposes to (n, c, h_out, w_out, kH, kW). Index arrays r and c_idx map each (oh, ow, kh, kw) combination to the (row, col) it corresponds to in the original input (row = oh*stride + kh, same for col). scatter_add then accumulates each patch-element's gradient into that position, correctly summing overlapping contributions instead of overwriting them.
        """
        n, c, _, _ = self.data.shape
        sN, sC, sH, sW = self.data.strides

        # Create a strided view of the input: (n, h_out, w_out, c, kH, kW)
        # Points to the same memory - implements a different way to navigate it.
        strided = as_strided(
            self.data,
            shape=(n, h_out, w_out, c, kH, kW),
            strides=(sN, stride * sH, stride * sW, sC, sH, sW),
        )

        # Reshape to (n, h_out * w_out, c * kH * kW)
        col = strided.reshape(n, h_out * w_out, -1)

        out = self._create_results(col, self)
        out.op = "im2col"

        def _backward():
            assert out.grad is not None
            grad = xp.zeros_like(self.data)

            # grad shape: (n, h_out * w_out, c * kH * kW)
            # reversing the reshape done in the forward pass to allow transposing to the desired shape -
            grad_strided = out.grad.reshape(n, h_out, w_out, c, kH, kW)

            # Transpose to (n, c, h_out, w_out, kh, kw)
            grad_patches = grad_strided.transpose(0, 3, 1, 2, 4, 5)

            # Build explicit index arrays for where each patch-element lands in the input
            # r_offsets[oh, kh] = kh + oh * stride  (the input row this (kh, oh) pair maps to)
            oh = xp.arange(h_out)
            kh = xp.arange(kH)
            ow = xp.arange(w_out)
            kw = xp.arange(kW)

            # Broadcast to get every (kh, kw, oh, ow) combination
            r = (
                oh[:, None, None, None] * stride + kh[None, None, :, None]
            )  # (h_out, 1, kH, 1)
            c_idx = (
                ow[None, :, None, None] * stride + kw[None, None, None, :]
            )  # (1, w_out, 1, kW)

            # cupyx's scatter_add handles the overlap correctly
            # Index Tuple: (n, c, (h_out, w_out, kh, kw)) ~ grad_patches.shape
            scatter_add(grad, (slice(None), slice(None), r, c_idx), grad_patches)

            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def pool2d(self, pool_size=2, stride=None):
        """
        Applies the max pooling operation on this tensor, reducing the overall size by choosing the maximum value within the given pool.

        Parameters
        ----------
        pool_size : int
            Size of the pooling kernel.
            Default is 2.
        stride : int
            Pool kernel stride.
            Default is None.

        Returns
        -------
        Tensor
            A new 4D tensor with reduced axes sizes.

        Notes
        -----
        Forward Pass:
            Constructs a 6D view of the input using as_strided(), shaped (n, h_out, w_out, c, kh, kw). The view does not copy data - it reinterprets the same underlying buffer by reading it with different strides, so overlapping patches share memory rather than duplicating it. xp.max() then reduces over the last two axes (the pooling window), leaving one max per value (n, c, oh, ow) window.

        Backward Pass:
            Only the maximum element of each window contributed to the forward output, so only that element should receive gradients; every other position in the window receives a zero.
            mask marks, for every position in every window, whether that position equals the window's max. Multiplying out.grad routes each window's gradient to its winning position and zeroes out the rest, producing grad_expanded in the windowed (n, c, h_out, w_out, pool_size, pool_size) shape.
            Index arrays r and c_idx map each (oh, ow, within-window) combination to the (row, col) it corresponds to in the original input.

        See Also
        --------
        im2col : Implementation of the unfold operation

        """
        n, c, h, w = self.data.shape
        if stride is None:
            stride = pool_size

        h_out = ((h - pool_size) // stride) + 1
        w_out = ((w - pool_size) // stride) + 1

        sN, sC, sH, sW = self.data.strides

        # Create strided view: (n, c, h_out, w_out, pool_size, pool_size)
        strided = as_strided(
            self.data,
            shape=(n, c, h_out, w_out, pool_size, pool_size),
            strides=(sN, sC, stride * sH, stride * sW, sH, sW),
        )

        # Forward: max over the pooling window axes (4, 5) - (pool_size, pool_size)
        out_data = xp.max(strided, axis=(4, 5))
        out = self._create_results(out_data, self)
        out.op = "pool2d"

        def _backward():
            assert out.grad is not None
            grad = xp.zeros_like(self.data)

            # Create the mask
            mask = strided == out_data[:, :, :, :, None, None]

            # (n, c, h_out, w_out, pool_size, pool_size)
            grad_expanded = out.grad[:, :, :, :, None, None] * mask

            # Build index arrays for where each pool element lands in the input
            oh = xp.arange(h_out)
            ow = xp.arange(w_out)
            p = xp.arange(pool_size)

            # (h_out, pool_size) → input row indices
            r = (
                oh[:, None, None, None] * stride + p[None, None, :, None]
            )  # (h_out, 1, pool_size, 1)
            c_idx = (
                ow[None, :, None, None] * stride + p[None, None, None, :]
            )  # (1, w_out, 1, pool_size)

            scatter_add(grad, (slice(None), slice(None), r, c_idx), grad_expanded)
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def relu(self):
        """
        Apply the Rectified Linear Unit (ReLU) activation function.

        Applies the element-wise operation f(x) = max(0, x).

        This is a non-linear activation that introduces sparsity in the network.

        Returns
        -------
        Tensor
            A new tensor with ReLU applied element-wise

        Notes
        -----
        Forward pass: Computes xp.maximum(0, self.data)

        Backward pass:
        Gradient is passed back only for elements where input > 0.
        For elements where input <= 0, gradient is zeroed.

        """
        op = xp.maximum(0, self.data)
        out = self._create_results(op, self)

        def _backward():
            self._accumulate_grad(out.grad * (out.data > 0))

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Apply the element-wise Sigmoid activation function.

        Squashes input values into a continuous range between 0 and 1.

        Mathematical Formula:
        f(x) = 1 / (1 + e^(-x))

        Returns
        -------
        Tensor
            A new Tensor containing the element-wise sigmoid activations.

        Notes
        -----
        Forward Pass: Computes the Sigmoid function element-wise for self.data
        Backward Pass:
        The derivative of the Sigmoid function simplifies to
        (forward_pass_output) * (1 - forward_pass_output)

        - Gradient w.r.t self: out.grad * (op * (1 - op))
        """
        x = self.data
        op = xp.where(x >= 0, 1.0 / (1.0 - xp.exp(-x)), 1.0 / (1.0 - xp.exp(x)))
        out = self._create_results(op, self)
        out.op = "sigmoid"

        def _backward():
            assert out.grad is not None
            local_grad = out.grad * (op * (1 - op))
            self._accumulate_grad(local_grad)

        out._backward = _backward
        return out

    def tanh(self):
        """
        Apply the element-wise Hyperbolic Tangent (Tanh) activation function.

        Squashes input values into a continuous range between -1 and 1.

        Mathematical formula:
        f(x) = (e^(x) - e^(-x)) / (e^(x) + e^(-x))

        Returns
        -------
        Tensor
            A new Tensor containing the element-wise tanh activations.

        Notes
        -----
        Forward Pass: (e^x - e^(-x)) / (e^x  + e^(-x))
        Backward Pass:
        The derivative of the tanhh function is as follows:
        1 - (forward_pass_output ^ 2)

        - Gradient w.r.t to self: out.grad * 1 - (op ** 2)
        """
        op = xp.tanh(self.data)
        out = self._create_results(op, self)
        out.op = "tanh"

        def _backward():
            assert out.grad is not None
            self._accumulate_grad(out.grad * (1.0 - (op**2)))

        out._backward = _backward
        return out

    def softmax(self):
        """
        Softmax activation function

        Applies the softmax function along the column axis

        Returns
        -------
        Tensor
            The activated tensor containing the probability distribution per row.

        Notes
        -----
        Computation graph automatically constructed via intermediate operations.
        """
        shifted = self - self.data.max(axis=1, keepdims=True)
        exp = shifted.exp()
        return exp / exp.sum(axis=1, keepdims=True)

    def mse(self, y):
        """
        Computes the Mean Square Error (MSE) loss against target values.

        Parameters
        ----------
        y : Tensor or array-like
            The ground truth target values matching this tensor's shape

        Returns
        -------
        Tensor
            A 0-dimensional (scalar) Tensor containing the average squared loss.

        Notes
        -----
        Computation graph automatically constructed via intermediate operations.
        Commonly used in regression tasks.
        """
        y = y if isinstance(y, Tensor) else Tensor(y)
        return (Tensor(0.5) * ((self - y) ** 2)).mean()

    def bce(self, y):
        """
        Computes the Binary Cross-Entropy (BCE) loss.

        Parameters
        ----------
        y : Tensor or array-like
            The true binary labels (0.0 or 1.0) matching this tensor's shape.

        Returns
        -------
        Tensor
            A 0-dimensional (scalar) Tensor containing the average BCE loss.

        Notes
        -----
        Computation graph automatically constructed via intermediate operations.
        Commonly used for binary classification tasks.
        Clips predictions internally to prevent numerical instability.
        """
        y = y if isinstance(y, Tensor) else Tensor(y)
        clipped = self.clip(1e-7, 1 - 1e-7)
        return -(
            y * clipped.log() + (Tensor(1.0) - y) * (Tensor(1.0) - clipped).log()
        ).mean()

    def ce(self, y):
        """
        Computes the Categorical Cross-Entropy (CE) loss.

        Parameters
        ----------
        y : Tensor or array-like
            The one-hot encoded ground truth probability target distributions.

        Returns
        -------
        Tensor
            A 0-dimensional (scalar) Tensor containing the average CE loss.

        Notes
        -----
        Computation graph automatically constructed via intermediate operations.
        Commonly used for multi-class classification tasks alongside a Softmax activation layer.
        Clips predictions internally to avoid numerical instability.
        """
        y = y if isinstance(y, Tensor) else Tensor(y)
        clipped = self.clip(1e-7, 1.0)
        return -(y * clipped.log()).sum(axis=1).mean()

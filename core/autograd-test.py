import numpy as np
from tensor import Tensor

a = Tensor(np.array([[1.0], [2.0], [3.0], [4.0]]), requires_grad=True)
b = Tensor(np.array([2.0]), requires_grad=True)

c = a + b

c.backward()

print("Data:", c.data)
print(f"Gradients:\na:{a.grad}\nb:{b.grad}\nc:{c.grad}")

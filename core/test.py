import numpy as np
from tensor import Tensor

x = Tensor(np.array([2.0, 3.0]))
w = Tensor(np.array([4.0, 5.0]))

y = (x**2.).sum()

y.backward()

print(x.grad)

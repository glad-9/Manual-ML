import numpy as np
from tensor import Tensor

x = Tensor(np.array([[2.0]]), requires_grad=True)
a = Tensor(np.array([[2.0]]), requires_grad=True)

y = x * 3
z = y.mean()
l = z - a

l.backward()

print(x.grad)
print(y.grad)
print(z.grad)
print(a.grad)

d = Tensor(np.array([1,2,3]), requires_grad=True)
e = 5 + d




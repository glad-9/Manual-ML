Status: Fixed
Component: core/tensor.py
Date: 2026-06-03

# Tensor requires_grad propagation bug

## Symptom

Training crashed during backpropagation with:

TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'

Location:
self._accumulate_grad(other.data * out.grad)

## Diagnosis
Intermediate tensors created during operations such as addition and
multiplication inherited requires_grad=False by default.

This above (and some variations of it) is one of the most common errors I received when trying to perform operations on my Tensors. 
Took me a while to figure out but the gist of the problem I realized was that my _accumulate_grad method did nothing if a Tensor had its requires_grad flag set to False.
The method existed to set any grads that were initialized to None to a zeros_like numpy array. Prior to this conditional was the check for the requires_grad flag. The issue was that many of the Tensors whose gradients I did wish to backpropagate had their requires_grad property set to False by default and never changed to True, thus being ignored in gradient accumulation and throwing errors for trying to operate on 'NoneTypes'. 

The resolution was in adding two things:
- A _create_results method that took in the data of the current Tensor and its parents to wrap the creation of the Out Tensor. 
    
    def _create_results(self, data, *parents):
        requires_grad = any(p.requires_grad for p in parents)

        out = Tensor(data, requires_grad=requires_grad)
        out._prev = set(parents)

        return out
    
    The crucial component here is how we check in all of the parents if any of them had the requires_grad flag set to true. If yes, then we set the Out Tensor's flag to be true as well, thus preventing any missing gradients.

- Secondly, I realized we were backpropagating through every single node regardless of whether it had requires_grad = True when calling the _backward method on the topologically sorted list of nodes. This is where I added the last check that would resolve the issue completely:

    for node in reversed(topo):
        if not node.requires_grad:
            continue
        node._backward()

    Essentially, we check whether a node has the requires_grad flag set to True or not and skip it if it doesn't. This prevented any unneeded gradients from entering the loop and being initialized to None. 

The print checks that enabled me to catch the bug:
- print(f"mul: out id={id(out)}, out.grad={out.grad.shape if hasattr(out.grad, 'shape') else out.grad}")
    Isolated the multiplication operation and its shape -> confirmed it was None
- print(f"calling backward on id: {id(node)}, shape: {node.data.shape if hasattr(node.data, 'shape') else node.data}, requires_grad: {node.requires_grad}")
    This was the smoking gun - essentially told me where exactly the process went wrong

    Output:
        calling backward on id: 139683741631744, shape: (1, 1), requires_grad: True
        mul: out id=139683741631744, out.grad=(1, 1)
        calling backward on id: 139683741632080, shape: (), requires_grad: False
        calling backward on id: 139683740895040, shape: (1, 1), requires_grad: True
        mean: out id=139683740895040, out.grad=[[-1.]]
        calling backward on id: 139683741632992, shape: (64, 1), requires_grad: True
        calling backward on id: 139683741631984, shape: (64, 1), requires_grad: True
        mul: out id=139683741631984, out.grad=(64, 1)
        calling backward on id: 139683741633424, shape: (64, 1), requires_grad: False
        calling backward on id: 139683746398576, shape: 1.0, requires_grad: False
        calling backward on id: 139683774425680, shape: (64, 1), requires_grad: False
        mul: out id=139683774425680, out.grad=None
        Traceback (most recent call last):


NOTE: Having grads initialized to None was important in resolving the issue because if this wasn't the case and we initialized to 0s instead, _accumulate_grad would not have caught it and placed a zero array grad for something we did not need. 

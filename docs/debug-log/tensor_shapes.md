Status: Fixed
Component: core/tensor.py
Date: 2026-06-03

# Broadcasting error in _accumulate_grad

## Symptom
Training crash during backpropagation due to non-broadcastable shapes:

ValueError: non-broadcastable output operand with shape (1,1) doesn't match the broadcast shape (64,1)

Location:

core/tensor.py", line 58, in _accumulate_grad
    self.grad += grad

## Diagnosis
The root cause for the error seems to come from trying to backpropagate the activation of the Linear layer - X @ W + b
The batch size in training is 64 and there is 1 neuron in the last layer and 8 from the previous.
By substituting the shapes, we have (64, 8) @ (8, 1) + (1, 1)
This results in (64, 1) + (1, 1)
But wait? This should be doable right? 
The issue is the order in which it was done because we were going backward: (1, 1) += (64, 1) [Inferred from the error message]
If this was the other way around, numpy would have handled it smoothly through a proper broadcast - which it did during forward.

This gives us a clue on the potential solution. A broadcast operation in the inference involved adding the (1,1) b to each row of the (64,1) X @ W 
This is something we must take into account during backpropagation - accumulating the gradient of b across each row that it was broadcasted. 
We would need a way to reverse the effect that the broadcast had. 
The broadcast mentioned above was duplicated across axis 0, so we should probably sum up the grad in that direction when reversing.

## Treatment
To do this, we'd create a helper method:
def _unbroadcast(grad, shape):
    for axis, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad

When an axis has a dimension 1, we know it was broadcasted, so we sum up the grad along these axes.

There is also another potential case where the shape could end up being: (8,) or something similar, having lesser dimensions.
To handle this:
while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

This ensures that grad's shape is the same as shape. 

The method was implemented in the _accumulate_grad method and applies automatically to Tensors which could have potentially been broadcasted (if dim == 1)

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return

        grad = self._unbroadcast(grad, self.data.shape)
        
        if self.grad is None:
            self.grad = np.zeros_like(self.data).astype('float64')

        self.grad += grad
        assert self.grad.shape == self.data.shape

## General Principle
Any tensor that was broadcast during forward requires gradient reduction back to original shape during backward.



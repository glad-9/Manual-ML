Status: Fixed
Component: nn/layers/conv/conv2d.py
Date: 2026-06-12

# None grad in conv layer

## Symptom
Training crash during backpropagation due to a None grad somewhere in the computation graph:

Location:
    Traceback (most recent call last):
      File "Manual ML/main.py", line 68, in <module>
        main()
      File "Manual ML/main.py", line 47, in main
        train_cost, cv_cost = model.fit(
                              ^^^^^^^^^^
      File "Manual ML/nn/network.py", line 60, in fit
        self.optimizer.step(self.layers)
      File "Manual ML/nn/optim/adam.py", line 24, in step
        p.grad += self.reg * p.data

TypeError: unsupported operand type(s) for +: 'NoneType' and 'float'

## Diagnosis
This occurred after I finished implementing my CNN module of this project. This time, it can be inferred from the error message that one of my weight matrices ended up being initialized with a None grad and didn't receive a backward propagation. I realized after a bit of trial and error that within my _im2col method of my conv2d class, I was creating a new tensor that was detached from the computation graph and was computed through X.data. This produced a disconnected leaf Tensor with no autograd edge back to X(the activation from the previous layer which should be tracked). 
Gradients flowed correctly through the flattened weight matrix but hit a dead end when I took the return of _im2col which was a numpy array casted to a Tensor but disconnected from the graph. 
I also added some print statements in the forward pass as well as the optimizer's step where it calculated p.grad to see where it was really failing. Here's the output
Conv2D (512, 8, 28, 28)
ReLU (512, 8, 28, 28)
MaxPool2D (512, 8, 14, 14)
Conv2D (512, 16, 14, 14)
ReLU (512, 16, 14, 14)
MaxPool2D (512, 16, 7, 7)
Flatten (512, 784)
Linear (512, 128)
ReLU (512, 128)
Dropout (512, 128)
Linear (512, 10)
Softmax (512, 10)
None grad on Conv2D, shape: (8, 1, 3, 3)

This confirmed the forward pass was working fine and the backward pass in layer 1 was where the error happened. 

## Treatment
To fix this issue, I moved im2col into tensor.py as a dedicated op with an explicit backward pass that scatters patch gradients back to their original spatial positions via +=

## General Principle
Any op that constructs a new array from tensor data using raw numpy produces a disconnected leaf. If this result needs to carry gradients, it should be implemented as a tracked tensor op.


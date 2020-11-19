import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
from Layer_dense import Layer_Dense as LD
nnfs.init()


# ReLU activation
class Activation_ReLU:

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from input
        self.output = np.maximum(0, inputs)

#create dataset
X, y = spiral_data(samples=100, classes =3)

# creae dense layer with 2 input features and 3 output values
dense1 = LD(2,3)

#create reLU activation (to be used with dense layer)
activation1 = Activation_ReLU()

#make a forward pass of our training date through this layer
dense1.forward(X)

#forward pass through activation func.
#takes in output from previous layer
activation1.forward(dense1.output)


#let's then have a look at the first few samples:
print(activation1.output[:5])


"""
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

#this is the ReLU function
output = []
for i in inputs:
    if i > 0:
        output.append(i)
    else:
        output.append(0)

print(output)


# or, in numpy, we can just use maximum
output = np.maximum(0, inputs)
print(output)
"""

"""
problem with step function: not very informative. It's hard to tell whether a
function was close to activating or not!

Thus, when it comes to optimizing weights and biases, it's easier for the
optimizer if we have activation functions that are more granular/informative

The original, more granular activation function is called the sigmoid function

y = 1 / (1 + e^(-x))

This function returns a value in the range of 0 (for negative infinity) through
.5 for the input of 0, and to 1 for positive infinity. We'll talk more abt this
later in chapter 16 :)

main attraction of neural networks is their ability to solve nonlinear problems

"""

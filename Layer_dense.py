import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        generate weights randomy
        this assumes we are not transfer ;earning, ofc
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        """
        initialize biases to zero, so that initially each neuron fires
        this can be a best practice, but there are also other techniques
        i.e. may be a bad idea to initialize all zeroes in the case of
        many dead neurons
        """
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs)
        self.output = np.dot(inputs, self.weights) + self.biases

#create a dense layer with two input features and three neurons
dense1 = Layer_Dense(2,3)

dense1.forward(X)

"""
np.random.randn -> a convenient way to initialize arrays
    Produces a gaussian distribution with a mean of 0 and a variance of 1
    In general, neural nets work best with values between -1 and +1
    We're multiplying by 0.01 to generate numbers that are a couple magnitudes
    smaller, so the model can take less time to fit the data during training
    (as otherwise starting vaues will be disproportionately large compared
    to the updates being made during training

    The idea here is to start a model with non-zero values small enough
    that they won't affect training. This way, we have a bunch of values
    to begin working with, but hopefully none too large and not too many 0s

np.random.randn(x,y) -> Takes dimension sizes as parameters and creates the
    output array with this shape. The weights here will be the number of
    neurons for the second dimension. This is similar to our previous made-
    up array of weights, just randomly generated.
"""

print(np.random.randn(2,3,4,3,2,4,2,5)
    



nnfs.init()
X,y = spiral_data(samples = 200, classes = 3)
plt.scatter(X[:,0], X[:,1],label="deez",c=y, cmap='brg')
#plt.ylabel(loc='bottom', fontsize=18)
plt.show()

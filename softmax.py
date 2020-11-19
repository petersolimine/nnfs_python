#softmax activation
#produces confidence scores for each class that add up to one
import numpy as np
from matplotlib import pyplot as plt
from Layer_dense import Layer_Dense as LD
from activation import Activation_ReLU
from nnfs.datasets import spiral_data

import nnfs
nnfs.init()

class Activation_Softmax:
    def forward(self, inputs):
        #get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)

        self.output = probabilities


#create dataset

X,y= spiral_data(samples=100,classes = 3)

#create dense layer with 2 input features and 3 output values
dense1= LD(2,3)

#create reLU activation (to be used with dense layer)
activation1= Activation_ReLU()

#create second dense layer with 3 input features  (as we take output
#of previous layer here) and 3 output values
dense2 = LD(3,3)

#create softmax activation (to be used with dense layer)
activation2 = Activation_Softmax()

#make forward pass of our training data through this layer
dense1.forward(X)

#make a forward pass through activation function
#it takes the output of second dense layer here
activation1.forward(dense1.output)

#make a forward pass through second dense layer
#it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

#make a firward pass through activation function
#it takes the output of second dense layer here
activation2.forward(dense2.output)

#let's see output of first few samples:
print('output oof first few samples:')
print(activation2.output[:5])



matrx = np.random.randn(2,3)
print(matrx)
#axis zero means sum down the columns, because our reference point is the rows
print(np.sum(matrx, axis=0))
print(np.sum(matrx, axis=1))
#axis = 0 means sum dowhn the columns 


layer_outputs = [4.5, 1.21, 2.475]

#for each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('exponentiated values:')
print(exp_values)

#normalize values
norm_values = exp_values / np.sum(exp_values)
print('normalized exponential values:')

print(norm_values)
print('sum of normalized values:', np.sum(norm_values))

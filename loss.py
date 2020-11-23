import numpy as np
from matplotlib import pyplot as plt
from Layer_dense import Layer_Dense as LD
from activation import Activation_ReLU
from nnfs.datasets import spiral_data
import nnfs
nnfs.init()

"""
cross entropy loss (not log loss, though it is a function of negative logs
'one hot' vectors i.e. [0,0,0,1,0,0,0]
"""

import math

# example of output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]
#ground truth
target_output = [1,0,0]

loss = -(math.log(softmax_output[0]*target_output[0] +
                  softmax_output[1]*target_output[1] +
                  softmax_output[2]*target_output[2]))

print(loss)

softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]

class_targets = [0, 1, 1]


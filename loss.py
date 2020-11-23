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

# print(loss)

"""
softmax_outputs = [[0.7, 0.1, 0.2],
                   [0.1, 0.5, 0.4],
                   [0.02, 0.9, 0.08]]
"""
class_targets = [0, 1, 1]

"""
for targ_idx, distribution in zip(class_targets, softmax_outputs):
    print(distribution[targ_idx])
"""
softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
average_loss = np.mean(neg_log)
print(average_loss)

if len(class_targets.shape) == 1:
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]
elif len(class_targets.shape) == 2:
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )

#losses
neg_log = -np.log(correct_confidences)

average_loss = np.mean(neg_log)
print(average_loss)

# remember, the limit of a natural logarithm of x, with x approaching zero from a positive, equals negative infinity
# it is impossible to calculate the natural logarithm of a negative value
# this is why clipping is important
# no matter what loss we use, we always take the mean to find overall loss


# common loss class
class Loss:

    # calculate the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        #calculate sample losses
        sample_losses = self.forward(output, y)

        #calculate mean loss
        data_loss = np.mean(sample_losses)

        #return loss
        return data_loss

#cross entropy loss
class Loss_CategoricalCrossentropy(Loss):

    #forward pass
    def forward(self, y_pred, y_true):

        #number of samples in a batch
        samples = len(y_pred)

        #clip data to prevent division by zero
        #clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # probabilities for target values
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = softmax_outputs[
                range(len(softmax_outputs)),
                class_targets
            ]

        # Mask values - only for one-hot encoded labels
        elif len(class_targets.shape) == 2:
            correct_confidences = np.sum(
                softmax_outputs * class_targets,
                axis=1
            )

        #losses
        negative_log_likelihoods = -np.sum(correct_confidences)
        return negative_log_likelihoods


"""
Important to remember the difference between loss and accuracy, which describes how often a model is correct
in terms of a fraction. Conveniently, we can reuse existing variable definitions to calculate the accuracy metric

"""
import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt
nnfs.init()

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
"""

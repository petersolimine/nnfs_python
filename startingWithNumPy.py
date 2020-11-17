import numpy as np
from nnfs.datasets import spiral_data
import nnfs
import matplotlib.pyplot as plt

nnfs.init()
x,y = spiral_data(samples = 200, classes = 3)
plt.scatter(x,y,title="hello")
plt.show()

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

#dot product gives one answer from multiplying element-wise and then summing
outputs = np.dot(weights, inputs) + biases

print(outputs)

#Where np.expand_dims() adds a new dimension at the index of the axis.
a=[1,2,3]
b=[2,3,4]
b=np.array([b])
a=np.expand_dims(np.array(a), axis=0).T

#b=b.T
z=np.dot(a,b)
print(z)
print()
print(a)
print(b)
print(np.dot(b,a))



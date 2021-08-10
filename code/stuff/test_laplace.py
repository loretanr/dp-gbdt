
import matplotlib.pyplot as plt
import numpy as np



f = open("laplace.txt", "r")
numbers = f.read().split(' ')
numbers = [float(bla) for bla in numbers[:-1]]


a = np.hstack(numbers)
_ = plt.hist(a, bins='auto')  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
# Text(0.5, 1.0, "Histogram with 'auto' bins")
plt.show()
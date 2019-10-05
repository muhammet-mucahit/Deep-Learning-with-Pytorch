import numpy as np

# A function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exps = np.exp(L)
    sum_of_exps = sum(exps)
    return [exp / sum_of_exps for exp in exps]

# Note: The function np.divide can also be used here, as follows:
# def softmax(L):
#     expL = np.exp(L)
#     return np.divide (expL, expL.sum())

L = np.array([1, 2, 3])
print(softmax(L))
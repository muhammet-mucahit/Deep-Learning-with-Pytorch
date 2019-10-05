import numpy as np

# A function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    return -np.sum([(Y_i * np.log(P_i)) + ((1 - Y_i) * np.log(1 - P_i)) for Y_i, P_i in zip(Y, P)])

# Other Solution
# def cross_entropy(Y, P):
#     Y = np.float_(Y)
#     P = np.float_(P)
#     return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

Y = [1, 0, 1, 1]
P = [0.4, 0.6, 0.1, 0.5]
print(cross_entropy(Y, P))
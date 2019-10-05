import numpy as np

#TODO Check correctness!!!

# A function that takes as input two lists Y, P,
# and returns the float corresponding to their multiclass_cross-entropy.
def multiclass_cross_entropy(Y, P):
    return -np.sum([y_ij * np.log(p_ij) for y, p in zip(Y, P) for y_ij, p_ij in zip(y, p)])

# Other Solution 1
# def multiclass_cross_entropy(Y, P):
#     Y = np.float_(Y)
#     P = np.float_(P)
#     return -np.sum(Y * np.log(P))

# Other Solution 2
# def multiclass_cross_entropy(Y, P):
#     Y = np.float_(Y)
#     P = np.float_(P)
#     return -np.sum(np.multiply(Y, np.log(P)))

Y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
]

P = [
    [0.7, 0.3, 0.1],
    [0.2, 0.4, 0.5],
    [0.1, 0.3, 0.4]
]
print(multiclass_cross_entropy(Y, P))
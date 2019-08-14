import numpy as np

# Helper functions

def notNone(lst):
    """ returns True if at least one item is not None """
    return sum([li is not None for li in lst]) > 0

def ceilzero(x):
    return max(x, 0)

def flatten(vec):
    return [val for sublist in vec for val in sublist]

def subsetmat(mat, inds):
    """ returns subset of a symmetric matrix, indexed by inds """
    subset = np.zeros((len(inds), len(inds)))
    for i in range(len(inds)):
        for j in range(len(inds)):
            subset[i, j] = mat[inds[i], inds[j]]
    return subset 

def cosinesim(v1, v2):
    if v1 is None or v2 is None:
        return 1
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

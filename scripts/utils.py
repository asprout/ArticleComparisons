import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 

# global variables
paywall_words = ["subscription", "subscribe", "full access", "digital access", "sign up", "unlimited access", 
            "unlimited digital access", "log in", "login", "sign up", "create an account", 
            "never miss a story", "for your first month", "to continue reading", "already a member",
            "rely on advertising", "click to continue"]

# Helper functions

def isalnum(str):
    # Returns true if any character of the string is alphanumeric
    for i in str:
        if i.isalnum(): 
            return True
    return False

def keywordsin(str, keywords = paywall_words):
    for word in keywords:
        if word in str:
            return True
    return False 

def readArticles(path):
    """ Reads df of articles from the given path, and adds a column to store the Doc 
    """
    article_df = pd.read_csv(path)
    article_df["doc"] = None 
    return article_df 


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

def prop_unique(vec, subset = None):
    if subset is not None:
        if len(subset) == 0:
            return None
        vec = [vec[i] for i in subset]
    return len(np.unique(vec))/len(vec)

def cosinesim(v1, v2):
    if v1 is None or v2 is None:
        return 1
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def display_mat(mat, normalize = False, xlabs = None, ylabs = None):
    """ Uses matplotlib to display mat; 
    if normalize is True, then coloring is normalized by range of values
    """
    # Only display a subset if matrix is too large
    mat = mat[0:min(50, mat.shape[0]), 0:min(50, mat.shape[1])]
    fig = plt.figure(figsize = (20, 20))
    ax = fig.add_subplot(111)
    if normalize:
        ax.matshow(mat, cmap = plt.cm.Blues)
    else:
        ax.matshow(mat, cmap = plt.cm.Blues, vmin = 0, vmax = 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, round(mat[i, j], 2), va = "center", ha = "center")
    ax.set_xticks([x for x in range(-1, mat.shape[1] + 1)])
    ax.set_yticks([x for x in range(-1, mat.shape[0] + 1)])
    if xlabs is not None: 
        ax.set_xticklabels([''] + xlabs)
        plt.xticks(rotation = 90)
        if ylabs is None and mat.shape[0] == mat.shape[1]:
            ylabs = xlabs
    if ylabs is not None:
        ax.set_yticklabels([''] + ylabs)

def minelapsed(start):
    return (time.time() - start)/60
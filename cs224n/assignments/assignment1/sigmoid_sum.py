#!/usr/bin/env python

import numpy as np


def sigmoid_sum(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    ### YOUR CODE HERE
    s = np.array((1 / (1 + np.exp(-x))).sum())
    ### END YOUR CODE

    return s


def sigmoid_sum_grad(x):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input is your original input x.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ### YOUR CODE HERE
    s = np.vectorize(sigmoid_sum)(x)
    ds = s * (1 - s)
    ### END YOUR CODE

    return ds


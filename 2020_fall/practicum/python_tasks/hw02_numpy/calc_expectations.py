import numpy as np


def add_mx(n, k):
    return np.triu(~np.triu(np.full((n, n), True), k))


def calc_expectations(h, w, X, Q):
    R = add_mx(Q.shape[0], h).T
    C = add_mx(Q.shape[1], w)
    return R.dot(Q).dot(C) * X

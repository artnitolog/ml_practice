import numpy as np


def euclidean_distance(X, Y):
    # faster than naive formula
    sqr = [(A ** 2).sum(axis=1) for A in [X, Y]]
    return np.sqrt(sqr[0][:, np.newaxis] + sqr[1] - 2 * np.inner(X, Y))


def cosine_distance(X, Y):
    # norm first, inner second - idea used in sklearn implementation
    cosine_sim = np.inner(X / np.linalg.norm(X, axis=1)[:, np.newaxis],
                          Y / np.linalg.norm(Y, axis=1)[:, np.newaxis])
    return 1 - cosine_sim


# Uniform interface for all metrics
_pw = {
    'euclidean': euclidean_distance,
    'cosine': cosine_distance,
}


def pw_distance(name, X, Y):
    return _pw[name](X, Y)

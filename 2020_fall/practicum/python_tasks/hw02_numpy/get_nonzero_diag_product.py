import numpy as np


def get_nonzero_diag_product(X):
    d = X.diagonal()
    idx = (d == 0)
    if idx.all():
        return None
    return d[~idx].prod()

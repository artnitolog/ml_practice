import numpy as np


def get_max_before_zero(x):
    y = x[1:][(x == 0)[:-1]]
    if y.size == 0:
        return None
    return y.max()

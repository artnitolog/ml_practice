import numpy as np


def encode_rle(x):
    if x.size == 0:
        return None
    idx = np.diff(np.insert(x, 0, x[0]-1)).nonzero()[0]
    return x[idx], np.diff(np.append(idx, len(x)))

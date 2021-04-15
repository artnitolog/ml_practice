import gc
import os
import numpy as np
from scipy.ndimage import shift, rotate, gaussian_filter
from joblib import Parallel, delayed


def _tr_parall(X, func, param):
    return Parallel(n_jobs=-1)(delayed(func)(im, param) for im in X)


_trs = {
    's': shift,
    'r': lambda im, ang: rotate(im, ang, reshape=False),
    'g': gaussian_filter,
}


def transform(X, transforms=None):
    """
    X: np.ndarray of flattened images
    transforms: list of tuples (trans, param)
                where trans is 's', 'r' or 'g'
    """
    if transforms is None:
        return X
    X_transformed = list(X.reshape((-1, 28, 28)))
    for tr, param in transforms:
        X_transformed = _tr_parall(X_transformed, _trs[tr], param)
    return np.array(X_transformed).reshape((len(X), -1))


def _parall_aug(X, trs, n_jobs_aug=1):
    if trs is None:
        return X
    step = len(X)
    X_aug = np.memmap('augmented_x', shape=(step * (len(trs)+1), X.shape[1]),
                      dtype=X.dtype, mode='w+')
    X_aug[0:step] = X

    def batch(i):
        print(step*(i+1), step*(i+2), trs[i])
        X_aug[step*(i+1):step*(i+2)] = transform(X, trs[i])
    Parallel(n_jobs=n_jobs_aug)(delayed(batch)(i) for i in
                                range(len(trs)))
    X_aug_out = np.copy(X_aug)
    del X_aug
    gc.collect()
    os.remove('augmented_x')
    return X_aug_out


def aug(X, trs):
    if trs is None:
        return X
    step = len(X)
    X_aug = np.empty(shape=(step * (len(trs)+1), X.shape[1]),
                     dtype=X.dtype)
    X_aug[0:step] = X
    for i in range(len(trs)):
        X_aug[step*(i+1):step*(i+2)] = transform(X, trs[i])
    return X_aug


def aug_X_y(X, y, trs):
    if trs is None:
        return X, y
    else:
        return aug(X, trs), np.tile(y, len(trs) + 1)

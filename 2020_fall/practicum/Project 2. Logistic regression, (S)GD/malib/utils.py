import numpy as np


def logloss(y, a):
    # y.shape == a.shape == (n, 2)
    return -((y * np.log(a)).sum(axis=-1).mean())


def accuracy(y, b):
    return (y == b).mean()


def grad_finite_diff(function, w, eps=1e-8):
    """
    Возвращает численное значение градиента,
    подсчитанноепо следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    f_d = np.apply_along_axis(function, 1, w + np.diag(np.full(len(w), eps)))
    return (f_d - function(w)) / eps

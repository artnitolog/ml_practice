import numpy as np
import scipy
from scipy.special import expit as sigmoid


class BaseSmoothOracle:
    """
    Useless class
    """

    def func(self, w):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    def __init__(self, l2_coef=0, fit_intercept=True):
        self.lambda_ = l2_coef
        self.lambda_half = l2_coef / 2
        self.fit_intercept = fit_intercept

    def func(self, X, y, w, intercept=None):
        if self.fit_intercept:
            ins = np.clip(sigmoid(y * (X.dot(w) + intercept)), 1e-20, 1-1e-20)
        else:
            ins = np.clip(sigmoid(y * (X.dot(w))), 1e-20, 1 - 1e-20)
        return -np.log(ins).mean() + self.lambda_half * np.inner(w, w)

    def grad(self, X, y, w, intercept=None):
        if self.fit_intercept:
            ins = y * sigmoid(-y * (X.dot(w) + intercept))
        else:
            ins = y * sigmoid(-y * (X.dot(w)))
        w_grad = -(X.T.dot(ins)) / X.shape[0] + w * self.lambda_
        if self.fit_intercept:
            return w_grad, -np.mean(ins)  # + self.lambda_ * intercept
        else:
            return w_grad

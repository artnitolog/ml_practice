import numpy as np
import scipy
from time import time
from scipy.special import expit as sigmoid
from tqdm import trange
from .oracles import BinaryLogistic


def defgen(n, seed=0):
    rng = np.random.default_rng(seed=seed)
    lim = 1 / (2 * n)
    return rng.uniform(-lim, lim, size=n)


class GDClassifier:
    def __init__(self, loss_function='binary_logistic',
                 step_alpha=1, step_beta=0, tolerance=1e-5,
                 max_iter=1000, fit_intercept=True, **kwargs):
        if loss_function != 'binary_logistic':
            raise ValueError('Ледяной горою айсберг')
        else:
            if fit_intercept:
                kwargs['fit_intercept'] = True
            self.oracle = BinaryLogistic(**kwargs)
        self.alpha = step_alpha
        self.negbeta = -step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def fit(self, X, y, w_0=None, trace=True, intercept_0=0, val_set=None):
        if w_0 is None:
            w_0 = defgen(X.shape[1])
        times = [-time()]
        self.w = w_0
        self.intercept = intercept_0
        losses = [self.get_objective(X, y)]
        times[-1] += time()
        if val_set is not None:
            acc = [(self.predict(val_set[0]) == val_set[1]).mean()]
        for self.n_iter in trange(1, self.max_iter + 1):
            times.append(-time())
            self.step(X, y)
            losses.append(self.get_objective(X, y))
            times[-1] += time()
            if val_set is not None:
                acc.append((self.predict(val_set[0]) == val_set[1]).mean())
            if abs(losses[-1] - losses[-2]) < self.tol:
                break
        if trace:
            if val_set is None:
                return {'time': times, 'func': losses}
            else:
                return {'time': times, 'func': losses, 'accuracy': acc}

    def step(self, X, y):
        lr = self.get_lr()
        if self.fit_intercept:
            grad, d_intercept = self.get_gradient(X, y)
            self.w -= lr * grad
            self.intercept -= lr * d_intercept
        else:
            grad = self.get_gradient(X, y)
            self.w -= lr * grad

    def predict(self, X):
        return np.where(X.dot(self.w) + self.intercept > 0, 1, -1)

    def predict_proba(self, X):
        proba_pos = sigmoid(X.dot(self.w) + self.intercept)
        return np.c_[1 - proba_pos, proba_pos]

    def get_objective(self, X, y):
        if self.fit_intercept:
            return self.oracle.func(X, y, self.w, self.intercept)
        else:
            return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        if self.fit_intercept:
            return self.oracle.grad(X, y, self.w, self.intercept)
        else:
            return self.oracle.grad(X, y, self.w)

    def get_lr(self):
        return self.alpha * self.n_iter ** self.negbeta

    def get_weights(self):
        return self.w

    def get_intercept(self):
        return self.intercept


class SGDClassifier(GDClassifier):

    def __init__(self, loss_function='binary_logistic',
                 step_alpha=1, step_beta=0, tolerance=1e-5,
                 max_iter=1000, random_seed=153, batch_size=500,
                 **kwargs):
        super().__init__(loss_function, step_alpha, step_beta,
                         tolerance, max_iter, **kwargs)
        self.batch_size = batch_size
        self.seed = random_seed

    def fit(self, X, y, w_0=None, trace=True, log_freq=1, intercept_0=0, val_set=None):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.


        cnt > log_freq * len / batch_size

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if w_0 is None:
            w_0 = defgen(X.shape[1])
        times = [-time()]
        self.w = w_0
        self.intercept = intercept_0
        losses = [self.get_objective(X, y)]
        times[0] += time()
        if val_set is not None:
            acc = [(self.predict(val_set[0]) == val_set[1]).mean()]
        rng = np.random.default_rng(seed=self.seed)
        n_batches, rem = divmod(X.shape[0], self.batch_size)
        batch_cnt = [0, 0]
        threshold = log_freq * X.shape[0] / self.batch_size
        times.append(-time())
        for self.n_iter in trange(1, self.max_iter + 1):  # epochs
            perm = rng.permutation(X.shape[0])
            for i, inds in enumerate(np.split(perm[:X.shape[0]-rem], n_batches)):
                batch_cnt[-1] += 1
                self.step(X[inds], y[inds])
                if batch_cnt[-1] - batch_cnt[-2] > threshold:
                    losses.append(self.get_objective(X, y))
                    times[-1] += time()
                    if val_set is not None:
                        acc.append((self.predict(val_set[0]) == val_set[1]).mean())
                    if abs(losses[-1] - losses[-2]) < self.tol:
                        break
                    if self.n_iter != self.max_iter or i != n_batches - 1:
                        batch_cnt.append(batch_cnt[-1])
                        times.append(-time())
            if batch_cnt[-1] - batch_cnt[-2] > threshold:
                break
        if len(losses) < len(times):
            losses.append(self.get_objective(X, y))
            times[-1] += time()
            if val_set is not None:
                acc.append((self.predict(val_set[0]) == val_set[1]).mean())
        if trace:
            history = {
                'time': times,
                'func': losses,
                'epoch_num': list(np.multiply(batch_cnt, self.batch_size / X.shape[0])),
            }
            if val_set is not None:
                history['accuracy'] = acc
            return history

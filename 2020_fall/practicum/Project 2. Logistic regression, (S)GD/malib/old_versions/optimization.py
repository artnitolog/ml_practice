import numpy as np
import scipy
from time import time
from scipy.special import expit as sigmoid
from .oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss_function, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        step_alpha - float, параметр выбора шага из текста задания
        step_beta- float, параметр выбора шага из текста задания
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        max_iter - максимальное число итераций     
        **kwargs - аргументы, необходимые для инициализации   
        """
        if loss_function != 'binary_logistic':
            raise ValueError('Ледяной горою айсберг')
        else:
            self.oracle = BinaryLogistic(**kwargs)
        self.alpha = step_alpha
        self.negbeta = -step_beta
        self.tol = tolerance
        self.max_iter = max_iter

    def fit(self, X, y, w_0=None, trace=False, init_seed=0):
        """
        Обучение метода по выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        w_0 - начальное приближение в методе
        trace - переменная типа bool
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            # np.random.seed(init_seed)
            w_0 = np.zeros(X.shape[1])
        times = [time()]
        self.w = w_0
        losses = [self.get_objective(X, y)]
        times[-1] -= time()
        for k in range(self.max_iter):
            times.append(time())
            self.w -= self.get_lr(k) * self.get_gradient(X, y)
            losses.append(self.get_objective(X, y))
            times[-1] -= time()
            if abs(losses[-1] - losses[-2]) < self.tol:
                break
        if trace:
            return {'time': times, 'func': losses}

    def predict(self, X):
        """
        Получение меток ответов на выборке X
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: одномерный numpy array с предсказаниями
        """
        return np.where(X.dot(self.w) > 0, 1, -1)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        return: двумерной numpy array, [i, k] значение соответветствует
        вероятности принадлежности i-го объекта к классу k
        """
        proba_pos = sigmoid(X.dot(self.w))
        return np.stack((1 - proba_pos, proba_pos))

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: float
        """
        return self.oracle.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_lr(self, k):
        return self.alpha * (k + 1) ** self.negbeta

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w

# class SGDClassifier(GDClassifier):
#     """
#     Реализация метода стохастического градиентного спуска для произвольного
#     оракула, соответствующего спецификации оракулов из модуля oracles.py
#     """

#     def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
#                  tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
#         """
#         loss_function - строка, отвечающая за функцию потерь классификатора. 
#         Может принимать значения:
#         - 'binary_logistic' - бинарная логистическая регрессия

#         batch_size - размер подвыборки, по которой считается градиент

#         step_alpha - float, параметр выбора шага из текста задания

#         step_beta- float, параметр выбора шага из текста задания

#         tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
#         Необходимо использовать критерий выхода по модулю разности соседних значений функции:
#         если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 


#         max_iter - максимальное число итераций (эпох)

#         random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
#         Этот параметр нужен для воспроизводимости результатов на разных машинах.

#         **kwargs - аргументы, необходимые для инициализации
#         """
#         pass

#     def fit(self, X, y, w_0=None, trace=False, log_freq=1):
#         """
#         Обучение метода по выборке X с ответами y

#         X - scipy.sparse.csr_matrix или двумерный numpy.array

#         y - одномерный numpy array

#         w_0 - начальное приближение в методе

#         Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
#         о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
#         превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
#         после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
#         Приближённый номер эпохи:
#             {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

#         log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
#         Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
#         будет превосходить log_freq.

#         history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
#         history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
#         history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
#         history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
#         (0 для самой первой точки)
#         """
#         pass

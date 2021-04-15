import os
import gc
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from .distances import pw_distance
from .augmentation import aug, aug_X_y, transform


class KNNClassifier:
    eps = 10 ** -5

    def __init__(self, k, strategy='my_own',
                 metric='euclidean', weights=False,
                 test_block_size=0, n_jobs=1, aug=None, tta=None):
        self.k = k
        self.strategy = strategy
        self._delegated = None
        if strategy != 'my_own':
            delegated_params = {
                'n_neighbors': k,
                'algorithm': strategy,
                'metric': metric,  # cosine will not work with trees
                'n_jobs': n_jobs,  # memory issues if > 1
            }
            self._delegated = NearestNeighbors(**delegated_params)
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size
        self.aug = aug
        self.tta = tta
        self.n_jobs = n_jobs

    def _get_pws(self, X):
        pws = pw_distance(self.metric, X, self.X_train)
        if self.tta is None:
            return pws
        else:
            for tr in self.tta:
                pws = np.minimum(pws, pw_distance(self.metric,
                                                  transform(X, tr),
                                                  self.X_train))
                gc.collect()
            return pws

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_train, self.y_train = aug_X_y(X, y, self.aug)
        if self.strategy != 'my_own':
            self._delegated.fit(self.X_train)

    def _find_kneighbors_batch(self, X, return_distance=False):
        if self.strategy == 'my_own':
            pws = self._get_pws(X)
            inds = pws.argsort(axis=1)[:, :self.k]
            if return_distance:
                # advanced indexing, instead of newer take_along_axis :(
                return pws[(np.arange(len(inds))[:, np.newaxis], inds)], inds
            else:
                return inds
        else:
            return self._delegated.kneighbors(X,
                                              return_distance=return_distance)

    def _fkn_naive(self, X, return_distance=False):
        size = self.test_block_size
        if size > 1 and (self.strategy not in ['kd_tree', 'ball_tree']):
            if return_distance:
                all_dists = np.empty((len(X), self.k), float)
                all_inds = np.empty((len(X), self.k), int)
                for i in range(0, len(X), size):
                    all_dists[i:i+size], all_inds[i:i+size] =\
                        self._find_kneighbors_batch(X[i:i+size], True)
                return all_dists, all_inds
            else:
                out = np.empty((len(X), self.k), dtype=int)
                for i in range(0, len(X), size):
                    out[i:i+size] = self._find_kneighbors_batch(X[i:i+size])
                return out
        else:
            return self._find_kneighbors_batch(X, return_distance)

    def _fkn_parall(self, X, return_distance=False):
        size = self.test_block_size
        if size > 1 and (self.strategy not in ['kd_tree', 'ball_tree']):
            if return_distance:
                dists_mm = np.memmap('fkn_dists_tmp', shape=(len(X), self.k),
                                     dtype=float, mode='w+')
                inds_mm = np.memmap('fkn_inds_tmp', shape=(len(X), self.k),
                                    dtype=int, mode='w+')

                def batch(X, i):
                    dists_mm[i:i+size], inds_mm[i:i+size] =\
                        self._find_kneighbors_batch(X[i:i+size], True)
                Parallel(n_jobs=self.n_jobs)(
                    delayed(batch)(X, i) for i in range(0, len(X), size)
                )
                all_dists = np.copy(dists_mm)
                all_inds = np.copy(inds_mm)
                del dists_mm
                del inds_mm
                gc.collect()
                os.remove('fkn_dists_tmp')
                os.remove('fkn_inds_tmp')
                return all_dists, all_inds
            else:
                i_mm = np.memmap('fkn_inds_tmp', shape=(len(X), self.k),
                                 dtype=int, mode='w+')

                def batch(X, i):
                    i_mm[i:i+size] = self._find_kneighbors_batch(X[i:i+size])
                Parallel(n_jobs=self.n_jobs)(
                    delayed(batch)(X, i) for i in range(0, len(X), size)
                )
                all_inds = np.copy(i_mm)
                del i_mm
                gc.collect()
                os.remove('fkn_inds_tmp')
                return all_inds
        else:
            return self._find_kneighbors_batch(X, return_distance)

    def find_kneighbors(self, X, return_distance=False):
        if self.n_jobs == 1:
            return self._fkn_naive(X, return_distance)
        else:
            return self._fkn_parall(X, return_distance)

    def _transform(self, X, return_weights=False):
        """
        Wrapper of find_kneighbors(), replacing:
            - indeces with classes,
            - distances with weights.
        """
        if return_weights:
            val = self.find_kneighbors(X, return_distance=True)
            return 1 / (val[0] + self.eps), self.y_train[val[1]]
        else:
            return self.y_train[self.find_kneighbors(X)]

    def predict(self, X):
        if self.weights:
            weights, all_neighs = self._transform(X, True)
            # map classes to indeces (bijection)
            poll = np.zeros((len(X), len(self.classes_)), dtype=weights.dtype)
            for i, class_ in enumerate(self.classes_):
                poll[:, i] = (weights * (all_neighs == class_)).sum(axis=1)
            return self.classes_[poll.argmax(axis=1)]
        else:
            all_neighs = self._transform(X)
            poll = np.zeros((len(X), len(self.classes_)), dtype=int)
            for i, class_ in enumerate(self.classes_):
                poll[:, i] = (all_neighs == class_).sum(axis=1)
            return self.classes_[poll.argmax(axis=1)]

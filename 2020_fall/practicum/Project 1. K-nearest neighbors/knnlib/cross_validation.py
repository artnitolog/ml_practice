import numpy as np
from time import time
from .nearest_neighbors import KNNClassifier


def kfold(n, n_folds=3):
    kf_list = []
    # Remainder is distributed among the first samples
    folds = np.array_split(np.arange(n), n_folds)
    for fold in folds:
        train = np.concatenate((np.arange(fold[0]),
                                np.arange(fold[-1] + 1, n)))
        kf_list += [(train, fold)]
    return kf_list


def _knn_cv_preds(clf, X_test, k_list):
    """
    Generator of knn-predictions depending on value in k_list.
    Yields tuple (k, y_pred).
    """
    classes_ = clf.classes_
    k_slices = zip(k_list[-2::-1], k_list[::-1])
    if clf.weights:
        clf.find_kneighbors(X_test, return_distance=True)
        weights, all_neighs = clf.find_kneighbors(X_test, return_distance=True)
        weights = 1 / (weights + 10 ** -5)
        all_neighs = clf.y_train[all_neighs]
        poll = np.zeros((len(X_test), len(classes_)), dtype=weights.dtype)
        for i, class_ in enumerate(classes_):
            poll[:, i] = (weights * (all_neighs == class_)).sum(axis=1)
        for lims in k_slices:
            yield lims[1], classes_[poll.argmax(axis=1)]
            idx = np.s_[:, lims[0]:lims[1]]
            for i, class_ in enumerate(classes_):
                poll[:, i] -= (weights[idx] *
                               (all_neighs[idx] == class_)).sum(axis=1)
        yield k_list[0], classes_[poll.argmax(axis=1)]
    else:
        all_neighs = clf.y_train[clf.find_kneighbors(X_test,
                                 return_distance=False)]
        poll = np.zeros((len(X_test), len(classes_)), dtype=np.int)
        for i, class_ in enumerate(classes_):
            poll[:, i] = (all_neighs == class_).sum(axis=1)
        for lims in k_slices:
            yield lims[1], classes_[poll.argmax(axis=1)]
            idx = np.s_[:, lims[0]:lims[1]]
            for i, class_ in enumerate(classes_):
                poll[:, i] -= (all_neighs[idx] == class_).sum(axis=1)
        yield k_list[0], classes_[poll.argmax(axis=1)]


def accuracy(x, y):
    return (x == y).mean()


# Uniform interface for all scores
_scores = {
    'accuracy': accuracy,
}


def get_score(name, x, y):
    return _scores[name](x, y)


def knn_cross_val_score(X, y, k_list=None, score='accuracy',
                        cv=None, return_times=False, **kwargs):
    if cv is None:
        cv = kfold(len(X))
    if k_list is None:
        k_list = [1, 3, 5]
    cv_dict = {k: np.empty(len(cv), dtype=float) for k in k_list}
    if return_times:
        times = {k: np.empty(len(cv), dtype=float) for k in k_list}
    for i, (train, test) in enumerate(cv):
        clf = KNNClassifier(k_list[-1], **kwargs)
        clf.fit(X[train], y[train])
        start = time()
        for k, y_pred in _knn_cv_preds(clf, X[test], k_list):
            cv_dict[k][i] = get_score(score, y_pred, y[test])
            finish = time()
            if return_times:
                times[k][i] = finish - start
            start = time()
    if return_times:
        return cv_dict, times
    else:
        return cv_dict

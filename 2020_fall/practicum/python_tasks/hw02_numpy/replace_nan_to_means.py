import numpy as np


def replace_nan_to_means(X):
    nanmeans = np.nan_to_num(np.nanmean(X, axis=0))
    return np.where(np.isnan(X), nanmeans, X)

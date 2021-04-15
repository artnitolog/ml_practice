# %%
import numpy as np


def _take_lists_arrays(idx_gen):
    def wrap(self, method='indexing'):
        inds = idx_gen(self)
        if method == 'indexing':
            for ind in inds:
                yield [obj[ind] for obj in self.objs]
        elif method == 'list_compr':
            for ind in inds:
                yield [[obj[i] for i in ind] for obj in self.objs]
    return wrap


class BatchGenerator:
    def __init__(self, list_of_sequences,
                 batch_size, shuffle=False):
        self.objs = list_of_sequences
        self.len = len(list_of_sequences[0])
        self.batch_size = batch_size
        self.shuffle = shuffle

    @_take_lists_arrays
    def _idx_gen(self):
        if self.shuffle:
            np.random.seed(0)
            inds = np.random.permutation(self.len)
            for i in range(0, self.len, self.batch_size):
                yield inds[i:i+self.batch_size]
        else:
            for i in range(0, self.len, self.batch_size):
                yield np.s_[i:i+self.batch_size]

    def __iter__(self):
        if isinstance(self.objs[0], list) and self.shuffle:
            return self._idx_gen(method='list_compr')
        else:
            return self._idx_gen()

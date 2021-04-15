# %%
import numpy as np


def encode_rle(x):
    if x.size == 0:
        return None
    idx = np.diff(np.insert(x, 0, x[0]-1)).nonzero()[0]
    return x[idx], np.diff(np.append(idx, len(x)))


# %%
class RleSequence():
    def __init__(self, input_sequence):
        self.numbers, self.counts = encode_rle(input_sequence)
        self._len = len(self.numbers)
        self._size = len(input_sequence)

    def __iter__(self):
        block = 0
        idx = 0
        while block != self._len:
            yield self.numbers[block]
            idx += 1
            if idx == self.counts[block]:
                block += 1
                idx = 0

    def __contains__(self, target):
        return target in self.numbers

    def _get_slice(self, start, stop, step):
        size = 1 + (stop - start - 1) // step
        # print(start, stop, step, size)
        out = np.empty(shape=max(size, 0), dtype=self.numbers.dtype)
        block = 0
        idx = start
        out_pos = 0
        while out_pos < size:
            while idx >= self.counts[block]:
                idx -= self.counts[block]
                block += 1
            out[out_pos] = self.numbers[block]
            out_pos += 1
            idx += step
        return out

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._get_slice(*i.indices(self._size))
        elif isinstance(i, int):
            if i < 0:
                i += self._size
            block = 0
            while i >= self.counts[block]:
                i -= self.counts[block]
                block += 1
            return self.numbers[block]
        else:
            raise IndexError

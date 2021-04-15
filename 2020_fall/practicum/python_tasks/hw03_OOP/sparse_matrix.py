from numbers import Integral
from copy import copy


def _check_idx(idx, shape=None):
    if not (isinstance(idx, tuple) and len(idx) == 2
            and isinstance(idx[0], Integral)
            and isinstance(idx[1], Integral)):
        return False
    if shape:
        return idx >= (0, 0) and idx[0] < shape[0] and idx[1] < shape[1]
    else:
        return True


class CooSparseMatrix:
    def __init__(self, ijx_list, shape, dict_no_check=None):
        if dict_no_check:
            self._shape = shape
            self.data = dict_no_check
            return
        if not (_check_idx(shape) and shape[0] > 0 and shape[1] > 0):
            raise TypeError
        self._shape = shape
        if ijx_list == []:
            self.data = dict()
            return
        for ijx in ijx_list:
            if not (isinstance(ijx, tuple) and len(ijx) == 3
                    and _check_idx(ijx[:2], shape)):
                raise TypeError
        ij = [(i, j) for i, j, _ in ijx_list]
        if len(ij) != len(set(ij)):
            raise TypeError
        self.data = {(i, j): x for (i, j, x) in ijx_list}

    def __getitem__(self, idx):
        if _check_idx(idx, self._shape):
            if idx in self.data:
                return self.data[idx]
            else:
                return 0
        elif _check_idx((idx, 0), self._shape):
            ijx_list = [(0, i[1], self.data[i]) for i in self.data
                        if i[0] == idx]
            return CooSparseMatrix(ijx_list, (1, self._shape[1]))
        else:
            raise TypeError

    def __setitem__(self, idx, val):
        if _check_idx(idx, self._shape):
            if val == 0 and idx in self.data:
                del self.data[idx]
            elif val != 0:
                self.data[idx] = val
        else:
            raise TypeError

    def __mul__(self, other):
        if other != 0:
            return CooSparseMatrix([], self._shape, {i: self.data[i] * other
                                   for i in self.data})
        else:
            return CooSparseMatrix([], self._shape)

    __rmul__ = __mul__

    def __add__(self, other):
        if self._shape != other._shape:
            raise TypeError
        if not self.data:
            return copy(other)
        if not other.data:
            return copy(self)
        new_dict = dict()
        for key in self.data:
            if key in other.data:
                val = self.data[key] + other.data[key]
                if val != 0:
                    new_dict[key] = val
            else:
                new_dict[key] = self.data[key]
        new_dict.update({key: other.data[key]
                        for key in other.data if key not in self.data})
        return CooSparseMatrix([], self._shape, new_dict)

    def __sub__(self, other):
        return -1 * other + self

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if not (_check_idx(val) and val[0] > 0 and val[1] > 0):
            raise TypeError
        if (val[0] * val[1] != self._shape[0] * self._shape[1]):
            raise TypeError
        new_dict = dict()
        for i, j in self.data:
            idx_fl = self._shape[1] * i + j
            new_dict[(idx_fl // val[1], idx_fl % val[1])] = self.data[(i, j)]
        self.data = new_dict
        self._shape = val

    @property
    def T(self):
        tr_data = dict()
        for idx in self.data:
            tr_data[idx[::-1]] = self.data[idx]
        return CooSparseMatrix([], self._shape[::-1], tr_data)

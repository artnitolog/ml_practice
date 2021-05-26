#!/usr/bin/python3

import sys
import numpy as np
from collections import defaultdict


def cosine_sim(u, v):
    return np.dot(u, v) / np.sqrt(np.square(u).sum() * np.square(v).sum())


class CommonRating:
    def __init__(self):
        self.i = []
        self.j = []

    def sim(self):
        if len(self.i) == 1:
            return int(self.i[0] * self.j[0] > 0)
        return cosine_sim(self.i, self.j)


def reduce(i, vals):
    pairs = defaultdict(CommonRating)
    for val in vals:
        rating_i, items, ratings = val.split('#')
        rating_i = float(rating_i)
        items = items.split(',')
        ratings = ratings.split(',')
        for j, rating in zip(items, ratings):
            pairs[j].j.append(float(rating))
            pairs[j].i.append(rating_i)
    items, sims = [], []
    for j, pair in pairs.items():
        sim = pair.sim()
        if sim > 0:
            items.append(j)
            sims.append(str(round(sim, 6)))
    if len(items) > 0:
        print(f"{i}@{','.join(items)}#{','.join(sims)}")


prev_i, vals = None, []
for line in sys.stdin:
    # sys.stderr.write(f'line: {line}\n')
    i, val = line.split('@')
    if i != prev_i:
        reduce(prev_i, vals)
        prev_i, vals = i, []
    vals.append(val)
reduce(i, vals)

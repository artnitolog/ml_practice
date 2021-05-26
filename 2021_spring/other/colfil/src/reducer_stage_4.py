#!/usr/bin/python3

import sys
import numpy as np


def reduce(key, ratings, sims):
    if key is None:
        return
    pred = np.average(ratings, weights=sims)
    if pred > 0:
        print(f'{key}@{pred:.6f}')


prev_key, ratings, sims = None, [], []
for line in sys.stdin:
    key, val = line.rsplit('@', 1)
    rating, sim = map(float, val.split(','))
    if key != prev_key:
        reduce(prev_key, ratings, sims)
        prev_key, ratings, sims = key, [], []
    ratings.append(rating)
    sims.append(sim)
reduce(key, ratings, sims)

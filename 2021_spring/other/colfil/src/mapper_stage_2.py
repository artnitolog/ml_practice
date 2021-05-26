#!/usr/bin/python3

import sys
import numpy as np


for line in sys.stdin:
    u, i_r = line.split('@')
    items, ratings = i_r.split('#')
    ratings = np.fromstring(ratings, sep=',')
    ratings_centered = np.array(ratings, dtype=float)
    ratings_centered -= ratings_centered.mean()
    ratings_centered = ratings_centered.round(6).astype(str)
    i_r_new = f"{items}#{','.join(ratings_centered)}"
    for i, r in zip(items.split(','), ratings_centered):
        print(f'{i}@{r}#{i_r_new}')

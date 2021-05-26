#!/usr/bin/python3

import sys
import numpy as np


def reduce(k, vals):
    u_s, r_s = [], []
    i_s, sim_s = [], []
    for val in vals:
        tag, n, val = val.split(',')
        if tag == 'r':
            u_s.append(n)
            r_s.append(int(val))
        elif tag == 's':
            i_s.append(n)
            sim_s.append(float(val))
    for u, r in zip(u_s, r_s):
        for i, sim in zip(i_s, sim_s):
            print(f'{u}@{i}@{r if i != k else -np.inf},{sim}')


prev_k, vals = None, []
for line in sys.stdin:
    k, val = line.strip().split('@')
    if k != prev_k:
        reduce(prev_k, vals)
        prev_k, vals = k, []
    vals.append(val)
reduce(k, vals)

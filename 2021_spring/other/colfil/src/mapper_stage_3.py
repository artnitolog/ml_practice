#!/usr/bin/python3

import sys
import os


for line in sys.stdin:
    if '@' in line:
        i, line = line.strip().split('@')
        for k, sim in zip(*map(lambda str_: str_.split(','), line.split('#'))):
            print(f'{k}@s,{i},{sim}')
    else:
        u, k, r, _ = line.split(',')
        if u == 'userId':
            continue
        print(f'{k}@r,{u},{float(r) * 2:.0f}')

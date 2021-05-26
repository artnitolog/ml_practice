#!/usr/bin/python3

import sys


for line in sys.stdin:
    u, i, r_ui, _ = line.strip().split(',')
    if u == 'userId':
        continue
    r_ui = float(r_ui) * 2  # 1, 2, ..., 10
    print(f'{u}@{i}#{r_ui:.0f}')

#!/usr/bin/python3

import sys

prev_user, items, ratings = None, [], []

for line in sys.stdin:
    # sys.stderr.write(f'line: {line}\n')
    user, val = line.strip().split('@')
    item, rating = val.split('#')
    if user != prev_user:
        if prev_user is not None:
            print(f"{prev_user}@{','.join(items)}#{','.join(ratings)}")
        prev_user, items, ratings = user, [], []
    items.append(item)
    ratings.append(rating)

print(f"{user}@{','.join(items)}#{','.join(ratings)}")
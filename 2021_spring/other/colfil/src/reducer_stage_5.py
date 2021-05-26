#!/usr/bin/python3

import sys


def format_pair(title, rating):
    rating = float(rating) / 10
    return f'{rating:.6f}#{title}'


prev_user, top100 = None, []

for line in sys.stdin:
    user, title, rating, _ = line.split('@')
    if user == prev_user:
        if len(top100) == 100:
            continue
        top100.append(format_pair(title, rating))
        if len(top100) == 100:
            print(f'{prev_user}@{"@".join(top100)}')
    else:
        if len(top100) < 100:
            assert prev_user is None, f'Not enough ratings oO, user {user}'
        prev_user, top100 = user, [format_pair(title, rating)]

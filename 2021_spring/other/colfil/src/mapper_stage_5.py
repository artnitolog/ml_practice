#!/usr/bin/python3

import sys
import pandas as pd

movie2title = pd.read_csv('movies.csv', index_col=0, usecols=[0, 1]).title

for line in sys.stdin:
    user, movie, rating = line.strip().split('@')
    print(f'{user}@{movie2title[int(movie)]}@{rating}@')

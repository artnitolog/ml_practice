#!/usr/bin/python3

import sys
import os

sys.stderr.write(f'Hi! Current input file: {os.environ["mapreduce_map_input_file"]}\n')

# Identity mapper
for line in sys.stdin:
    print(line.strip())

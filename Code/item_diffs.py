import math
import numpy as np
from collections import deque
from copy import deepcopy
import random
# from tqdm import tqdm  # <-- Change this import
import sys
from common import *

from multiprocessing import Pool
from collections import defaultdict
import tqdm
import json

# Computes item differences and also weighted item diffs w.r.t item similarity


def compute_item_diffs():
    item_diffs = defaultdict(dict)
    weighted_item_diffs = defaultdict(dict)
    valid_items = set(TOP_100_ITEM_SIMS.keys())
    COMMON_ITEM = 363
    # Similarities score
    for n in tqdm.tqdm(list(RATED_BY[COMMON_ITEM])):
        for o_n in RATED_BY[COMMON_ITEM]:
            rated_n = set(RATINGS[n].keys()).intersection(valid_items)
            rated_o_n = set(RATINGS[o_n].keys()).intersection(valid_items)

            weighted_sums = defaultdict(float)
            ir_diff = defaultdict(float)
            for m1 in rated_n:
                for m2 in rated_o_n:
                    if m1 == m2:
                        continue
                    weighted_sums[m1] += (1-(abs(RATINGS[n][m1] -
                                                 RATINGS[o_n][m2])/4)) * TOP_100_ITEM_SIMS[m1][m2]
                weighted_sums[m1] /= max(len(rated_o_n), 1)

            for vi in ORDERED_ITEMS[:10]:
                count = 0
                for i in rated_n.intersection(rated_o_n):
                    ir_diff[vi] += 1 - \
                        (abs(RATINGS[n][i] - RATINGS[o_n][i])/4)
                    count += 1
                ir_diff[vi] /= max(count, 1) * 2
                if count != 0:
                    ir_diff[vi] += (len(rated_n.intersection(rated_o_n) -
                                    {vi})/10) / 2

            weighted_item_diffs[int(n)][int(o_n)] = dict(weighted_sums)
            item_diffs[int(n)][int(o_n)] = dict(ir_diff)

    return weighted_item_diffs, item_diffs


if __name__ == '__main__':
    weighted_item_diffs, item_diffs = compute_item_diffs()
    with open('weighted_item_diffs_new.json', 'w') as f:
        f.write(json.dumps(weighted_item_diffs))
    with open('item_diffs_new.json', 'w') as f:
        f.write(json.dumps(item_diffs))

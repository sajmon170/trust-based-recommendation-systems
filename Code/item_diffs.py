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
            
            if len(rated_n) == 0 or len(rated_o_n) == 0:
                item_diffs[n][o_n] = 0
                weighted_item_diffs[n][o_n] = 0
            else:
                weighted_diff_sum = 0
                item_diff_sum = 0
                division_factor = 0
                for m1 in rated_n:
                    if m1 in rated_o_n:
                        division_factor += 1
                        item_diff_sum += (1-(abs(RATINGS[n][m1] - RATINGS[o_n][m1])/4))
                    for m2 in rated_o_n:
                        weighted_diff_sum += (1-(abs(RATINGS[n][m1] - RATINGS[o_n][m2])/4)) * TOP_100_ITEM_SIMS[m1][m2]
                        
                    
                weighted_item_diffs[n][o_n] = weighted_diff_sum / max(division_factor, 1)
                item_diffs[n][o_n] = item_diff_sum / (len(rated_n) * len(rated_o_n))
                
    return weighted_item_diffs, item_diffs

if __name__ == '__main__':
    weighted_item_diffs, item_diffs = compute_item_diffs()
    with open('weighted_item_diffs.json', 'w') as f:
        f.write(json.dumps(weighted_item_diffs))
    with open('item_diffs_new.json', 'w') as f:
        f.write(json.dumps(item_diffs))
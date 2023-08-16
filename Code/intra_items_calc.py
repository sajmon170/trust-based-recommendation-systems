import math
import numpy as np
from collections import deque
from copy import deepcopy
import random
# from tqdm import tqdm  # <-- Change this import
import sys
from common import *

from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map

# Returns the set of common users who have rated both movie i and j
def UC(i, j):
    return RATED_BY[i].intersection(RATED_BY[j])
# Returns the mean average rating given by a user
def r_(u):
    return sum(list(RATINGS[u].values())) / len(RATINGS[u])
# Returns the Pearson correlation of items i and j
def corr(i, j):
    uc = UC(i, j)
    
    def _numerator_term(u, i, j):
        mean = r_(u)
        return (RATINGS[u][i] - mean) * (RATINGS[u][j] - mean)
    
    def _denominator_term(u, i):
        return pow(RATINGS[u][i] - r_(u), 2)
    
    numerator = sum((_numerator_term(u, i, j) for u in uc))
    denominator_1 = math.sqrt(sum((_denominator_term(u, i) for u in uc)))
    denominator_2 = math.sqrt(sum((_denominator_term(u, i) for u in RATED_BY[i])))
    
    return numerator / (denominator_1 * denominator_2 + 1e-4) # Add a small delta to avoid division by zero
def sim(i, j):
    uc = UC(i, j)
    s = (1/(1 + pow(math.e, -len(uc)/2))) * corr(i,j)
    return s

def sim_wrapper(args):
    return sim(*args)

# Use a multiprocessing pool to parallelize the calculation of item_sims
def calculate_item_sims_parallel(item_id, rated_by_keys):
    total = len(rated_by_keys)
    item_sims = process_map(sim_wrapper, [(item_id, x) for x in rated_by_keys],
                            total=total, desc='Calculating item similarities', chunksize=1000, max_workers=8)
    return item_sims

if __name__ == '__main__':
    item_sims = np.array(calculate_item_sims_parallel(363, sorted(list(RATED_BY.keys()))))
    np.save('item_sims.npy', item_sims)

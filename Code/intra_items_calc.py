import math
from common import *

# Returns the set of common users who have rated both movie i and j


def UC(i, j, rated_by):
    return rated_by[i].intersection(rated_by[j])
# Returns the mean average rating given by a user


def r_(u, ratings):
    return sum(list(ratings[u].values())) / len(ratings[u])


def corr(i, j, ratings, rated_by):
    uc = UC(i, j, rated_by)

    def _numerator_term(u, i, j):
        mean = r_(u, ratings)
        return (ratings[u][i] - mean) * (ratings[u][j] - mean)

    def _denominator_term(u, i):
        return pow(ratings[u][i] - r_(u, ratings), 2)

    numerator = sum((_numerator_term(u, i, j) for u in uc))
    denominator_1 = math.sqrt(sum((_denominator_term(u, i) for u in uc)))
    denominator_2 = math.sqrt(
        sum((_denominator_term(u, i) for u in rated_by[i])))

    # Add a small delta to avoid division by zero
    return numerator / (denominator_1 * denominator_2 + 1e-4)


def sim(i, j, ratings, rated_by):
    uc = UC(i, j, rated_by)
    s = (1/(1 + pow(math.e, -len(uc)/2))) * corr(i, j, ratings, rated_by)
    return s


def jaccard_sim(i, j, _, rated_by):
    return len(rated_by[i].intersection(rated_by[j])) / len(rated_by[i].union(rated_by[j]))

# def sim_wrapper(args):
#     return sim(*args)

# # Use a multiprocessing pool to parallelize the calculation of item_sims
# def calculate_item_sims_parallel(item_id, rated_by_keys):
#     total = len(rated_by_keys)
#     item_sims = process_map(sim_wrapper, [(item_id, x) for x in rated_by_keys],
#                             total=total, desc='Calculating item similarities', chunksize=1000, max_workers=8)
#     return item_sims

# if __name__ == '__main__':
#     item_sims = np.array(calculate_item_sims_parallel(363, sorted(list(RATED_BY.keys()))))
#     np.save('item_sims.npy', item_sims)

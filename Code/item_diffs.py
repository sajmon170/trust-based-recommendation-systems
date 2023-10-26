from copy import deepcopy
from common import *


def item_diff_similarity(u1, u2, data, excl_item=None):
    rating = 0
    i1 = set(data['ratings'][u1].keys())
    i2 = set(data['ratings'][u2].keys())
    for i in i1.intersection(i2) - {excl_item}:
        rating += 1 - abs(data['ratings'][u1][i] - data['ratings'][u2][i])/4
    return rating / max(len(i1.intersection(i2) - {excl_item}), 1)


def weighted_item_diff_similarity(u1, u2, data, item_sims, target_item):
    rating = 0
    i1 = set(data['ratings'][u1].keys())
    i2 = set(data['ratings'][u2].keys())
    for i in (intr := (i1.intersection(i2) - {target_item})):
        i1_rating = data['ratings'][u1][i]
        i2_rating = data['ratings'][u2][i]
        i_j_sim = item_sims[target_item][i]

        rating += (1 - abs(i1_rating - i2_rating)/4)*i_j_sim
    return rating / max(len(intr), 1)

# def compute_item_diffs():
#     item_diffs = defaultdict(dict)
#     weighted_item_diffs = defaultdict(dict)
#     valid_items = set(TOP_100_ITEM_SIMS.keys())
#     COMMON_ITEM = 363
#     # Similarities score
#     for n in tqdm.tqdm(list(RATED_BY[COMMON_ITEM])):
#         for o_n in RATED_BY[COMMON_ITEM]:
#             rated_n = set(RATINGS[n].keys()).intersection(valid_items)
#             rated_o_n = set(RATINGS[o_n].keys()).intersection(valid_items)

#             weighted_sums = defaultdict(float)
#             ir_diff = defaultdict(float)
#             for m1 in rated_n:
#                 for m2 in rated_o_n:
#                     if m1 == m2:
#                         continue
#                     weighted_sums[m1] += (1-(abs(RATINGS[n][m1] -
#                                                  RATINGS[o_n][m2])/4)) * TOP_100_ITEM_SIMS[m1][m2]
#                 weighted_sums[m1] /= max(len(rated_o_n), 1)

#             for vi in ORDERED_ITEMS[:10]:
#                 count = 0
#                 for i in rated_n.intersection(rated_o_n):
#                     ir_diff[vi] += 1 - \
#                         (abs(RATINGS[n][i] - RATINGS[o_n][i])/4)
#                     count += 1
#                 ir_diff[vi] /= max(count, 1) * 2
#                 if count != 0:
#                     ir_diff[vi] += (len(rated_n.intersection(rated_o_n) -
#                                     {vi})/10) / 2

#             weighted_item_diffs[int(n)][int(o_n)] = dict(weighted_sums)
#             item_diffs[int(n)][int(o_n)] = dict(ir_diff)

#     return weighted_item_diffs, item_diffs


# if __name__ == '__main__':
#     weighted_item_diffs, item_diffs = compute_item_diffs()
#     with open('weighted_item_diffs_new.json', 'w') as f:
#         f.write(json.dumps(weighted_item_diffs))
#     with open('item_diffs_new.json', 'w') as f:
#         f.write(json.dumps(item_diffs))

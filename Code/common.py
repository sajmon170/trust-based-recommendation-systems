from utils import *
from intra_items_calc import sim
from collections import defaultdict
from tqdm.notebook import tqdm


def load_dataset(dataset_name: str, n_items=10):
    print(f"Loading dataset: {dataset_name}\nKeeping top {n_items} items.")

    # Load in epinions dataset
    trust_graph = {}  # Source user -> List of trusted users
    ratings = {}  # Source user -> {movie : +/- rating} map
    rated_by = {}  # Movie -> Set of users who have rated it
    item_jaccard = defaultdict(dict)
    jaccard_idx = defaultdict(dict)

    # Form trust graph
    with open(f'../datasets/{dataset_name}/trust_data.txt') as in_file:
        for l in in_file.readlines():
            f, t, _ = [int(v) for v in l.split()]
            if f not in trust_graph:
                trust_graph[f] = [t]
            else:
                trust_graph[f].append(t)
            if t not in trust_graph:
                trust_graph[t] = []

    # Map users to their ratings
    with open(f'../datasets/{dataset_name}/ratings_data.txt') as in_file:
        for l in in_file.readlines():
            s, m, r = l.split()
            s, m, r = int(s), int(m), float(r)
            if s not in ratings:
                ratings[s] = {m: r}
            else:
                ratings[s][m] = r

            if s not in trust_graph:
                trust_graph[s] = []

            # Fulfil rated_by graph
            if m not in rated_by:
                rated_by[m] = set([s])
            else:
                rated_by[m].add(s)

        for t in trust_graph:
            if t not in ratings:
                ratings[t] = {}

    # Top k items
    ordered_items = sorted(
        rated_by.keys(), key=lambda x: len(rated_by[x]), reverse=True)

    # Keep items
    items = ordered_items[:n_items]

    # Compute item similarities
    item_sims = {i: {j: sim(i, j, ratings, rated_by)}
                 for i in items for j in items}

    # Compute item jaccards
    for node in tqdm(rated_by[items[0]], "Computing jaccard indexes"):
        for other_node in rated_by[items[0]]:
            if node == other_node:
                continue
            item_jaccard[node][other_node] = item_jaccard_compute(
                ratings, node, other_node)
            jaccard_idx[node][other_node] = jaccard_index_neighbours(
                trust_graph, node, other_node)

    trust_graph_reduced, ratings_reduced, rated_by_reduced = remove_non_voters(
        trust_graph, ratings, rated_by, items)

    return {
        "trust_graph": trust_graph_reduced,
        "ratings": ratings_reduced,
        "rated_by": rated_by_reduced,
        "items": items,
        "item_sims": item_sims,
        "item_jaccard": item_jaccard,
        "jaccard_idx": jaccard_idx
    }


# Load in epinions dataset
DATA = {
    "epinions": load_dataset("epinions"),
    "filmtrust": load_dataset("filmtrust"),
}

# JACCARD_MATRIX = np.fromfile('jaccard.bin', dtype='float16').reshape(
#     (len(trust_graph), len(trust_graph)))
# ITEM_JACCARDS = np.fromfile('item_jaccard.bin',  dtype='float16').reshape(
#     (len(trust_graph), len(trust_graph)))

# with open('top_k_sims.json', 'r') as in_file:
#     TOP_100_ITEM_SIMS = json.load(in_file)
#     for k, i in list(TOP_100_ITEM_SIMS.items()):
#         new_i = {int(k1): v for k1, v in i.items()}
#         TOP_100_ITEM_SIMS[int(k)] = new_i
#         del TOP_100_ITEM_SIMS[k]

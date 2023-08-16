from utils import *
import json

# Load in epinions dataset
TRUST_GRAPH = {} # Source user -> List of trusted users
RATINGS = {} # Source user -> {movie : +/- rating} map
RATED_BY = {} # Movie -> Set of users who have rated it

# Form trust graph
with open('../datasets/epinions/trust_data.txt') as in_file:
    for l in in_file.readlines():
        f, t, _ = [int(v) for v in l.split()]
        if f not in TRUST_GRAPH:
            TRUST_GRAPH[f] = [t]
        else:
            TRUST_GRAPH[f].append(t)
        if t not in TRUST_GRAPH:
            TRUST_GRAPH[t] = []
            

# Map users to their ratings
with open('../datasets/epinions/ratings_data.txt') as in_file:
    for l in in_file.readlines():
        s, m, r = [int(v) for v in l.split()]
        if s not in RATINGS:
            RATINGS[s] = {m : r}
        else:
            RATINGS[s][m] = r

        if s not in TRUST_GRAPH:
            TRUST_GRAPH[s] = []
        
        # Fulfil rated_by graph
        if m not in RATED_BY:
            RATED_BY[m] = set([s])
        else:
            RATED_BY[m].add(s)
    
    for t in TRUST_GRAPH:
        if t not in RATINGS:
            RATINGS[t] = {}
            
# Top 5 items
ORDERED_ITEMS = sorted(RATED_BY.keys(), key=lambda x: len(RATED_BY[x]), reverse=True)

JACCARD_MATRIX = np.fromfile('jaccard.bin', dtype='float16').reshape((len(TRUST_GRAPH), len(TRUST_GRAPH)))
ITEM_DIFFS = np.fromfile('item_diffs.bin',  dtype='float16').reshape((len(TRUST_GRAPH), len(TRUST_GRAPH)))
ITEM_JACCARDS = np.fromfile('item_jaccard.bin',  dtype='float16').reshape((len(TRUST_GRAPH), len(TRUST_GRAPH)))
ITEM_DIFFS_SIM = csr_matrix(load_npz('item_sim_matrix.npz'))

with open('top_k_sims.json', 'r') as in_file:
    TOP_100_ITEM_SIMS = json.load(in_file)
    for k, i in list(TOP_100_ITEM_SIMS.items()):
        new_i = {int(k1): v for k1, v in i.items()}
        TOP_100_ITEM_SIMS[int(k)] = new_i
        del TOP_100_ITEM_SIMS[k]
        
with open('item_diffs_new.json', 'r') as in_file:
    ITEM_DIFFS_NEW = json.load(in_file)
    
with open('weighted_item_diffs.json', 'r') as in_file:
    WEIGHTED_ITEM_DIFFS = json.load(in_file)
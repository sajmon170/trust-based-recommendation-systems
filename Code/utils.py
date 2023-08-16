# Imports
import math
from collections import deque
import random
import sys
from copy import deepcopy
from tqdm.notebook import tqdm
import numpy as np
from scipy.sparse import load_npz, csr_matrix
 
sys.setrecursionlimit(10**6)

def evaluate(recommender, graph, ratings, rated_by, TEST_ITEM=386, TEST_PERCENTAGE=0.3):
    def _get_k(items, k):
        return random.sample(list(items), k=k)
    
    test_points = _get_k(rated_by[TEST_ITEM], math.ceil(len(rated_by[TEST_ITEM]) * TEST_PERCENTAGE))
    test_graph = deepcopy(graph)
    test_ratings = deepcopy(ratings)
    test_rated_by = deepcopy(rated_by)

    # Remove items from test set
    for tp in test_points:
        del test_ratings[tp][TEST_ITEM]
        test_rated_by[TEST_ITEM].remove(tp)
        
    # Compute recommendations
    recommendations = recommender(test_graph, test_ratings, test_rated_by, TEST_ITEM, test_points)

    # Compute accuracy metrics
    correct, incorrect, differenceSq, difference = 0, 0, 0, 0

    for tp in test_points:
        if round(recommendations[tp]) == ratings[tp][TEST_ITEM]:
            correct += 1
        else:
            incorrect += 1
        
        differenceSq += pow(recommendations[tp] - ratings[tp][TEST_ITEM], 2)
        difference += abs(recommendations[tp] - ratings[tp][TEST_ITEM])
            
    acc, mse, diff = correct / len(test_points), differenceSq / len(test_points), difference / len(test_points)
    return acc, mse, diff


def remove_non_voters(graph, ratings, rated_by, ITEMS : list[int]):
    keep_nodes = rated_by[ITEMS[0]]
    # for n in list(keep_nodes):
    #     if len(set(ratings[n].keys()).intersection(ITEMS)) < 2:
    #         keep_nodes.remove(n)# Those who have rated atleast 2 of our items
            
    reduced_graph = {}
    reduced_ratings = {}
    reduced_rated_by = {i: set([]) for i in ITEMS}

    for n in keep_nodes:
        reduced_graph[n] = list(set(graph[n]).intersection(keep_nodes))
        reduced_ratings[n] = {i : ratings[n][i] for i in ratings[n] if i in ITEMS}
        for i in reduced_ratings[n]:
            reduced_rated_by[i].add(n)
    
    return reduced_graph, reduced_ratings, reduced_rated_by



def create_jacc_connected_graph(graph, jaccards):
    new_graph = {}
    for n in graph:
        new_graph[n] = []
        for m in graph[n]:
            if n == m:
                continue
            if jaccards[n][m] > 0:
                new_graph[n].append((m, jaccards[n][m]))
    return new_graph

# A function used by DFS
def DFSUtil(graph, v,visited, curr_SCC):
    # Mark the current node as visited and print it
    visited.add(v)
    curr_SCC.append(v)
    # Recur for all the vertices adjacent to this vertex
    for i in graph[v]:
        if i not in visited:
            DFSUtil(graph, i,visited, curr_SCC)
 
def fillOrder(graph, v, visited, stack):
    # Mark the current node as visited
    visited.add(v)
    #Recur for all the vertices adjacent to this vertex
    for i in graph[v]:
        if i not in visited:
            fillOrder(graph, i, visited, stack)
    stack = stack.append(v)
     
 # Function that returns reverse (or transpose) of this graph
def getTranspose(graph):
    t_graph = {}
    # Recur for all the vertices adjacent to this vertex
    for i in graph:
        for j in graph[i]:
            if j not in t_graph:
                t_graph[j] = [i]
            else:
                t_graph[j].append(i)
            
            if i not in t_graph:
                t_graph[i] = []
    return t_graph
  
# The main function that finds and prints all strongly
# connected components
def getSCCs(graph):
    curr_SCC = []
    SCCs = []
    stack = []
    # Mark all the vertices as not visited (For first DFS)
    visited = set([])
    # Fill vertices in stack according to their finishing
    # times
    for i in graph:
        if i not in visited:
            fillOrder(graph, i, visited, stack)

    # Create a reversed graph
    gr = getTranspose(graph)
        
    # Mark all the vertices as not visited (For second DFS)
    visited = set([])

    # Now process all vertices in order defined by Stack
    while stack:
        i = stack.pop()
        if i not in visited:
            DFSUtil(gr, i, visited, curr_SCC)
            SCCs.append(curr_SCC)
            curr_SCC = []
    
    if curr_SCC != []:
        SCCs.append(curr_SCC)
    return SCCs

def can_reach_voter(graph, rated_by, node, item):
    q = deque([node])
    visited = set()
    target_nodes = rated_by[item]
    while len(q):
        current = q.popleft()
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current in target_nodes:
            return True
        
        q.extend((set(graph[current]) - visited) if current in graph else [])
    return False

def jaccard_index(graph, v1, v2):
    s1 = set(graph[v1])
    s2 = set(graph[v2])
    if len(s1) == len(s2) and len(s1) == 0:
        return 0
    jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
    return jaccard

def jaccard_index_neighbours(graph, v1, v2):
    ji = jaccard_index(graph, v1, v2)
    if v1 in graph[v2] or v2 in graph[v1]:
        ji += (1 / len(set(graph[v1]).union(set(graph[v2]))))
    return ji


def two_h_jaccard_index(graph, v1, v2):
    s1 = set(graph[v1])
    s2 = set(graph[v2])
    for k in list(s1):
        s1.update(graph[k])
    for k in list(s2):
        s2.update(graph[k])
    if len(s1) == len(s2) and len(s1) == 0:
        return 0
    jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
    return jaccard

# def compute_matricies():
#     item_diffs = np.zeros((len(TRUST_GRAPH), len(TRUST_GRAPH)), dtype='float16')
#     item_jaccards = np.zeros((len(TRUST_GRAPH), len(TRUST_GRAPH)), dtype='float16')
#     jaccards = np.zeros((len(TRUST_GRAPH), len(TRUST_GRAPH)), dtype='float16')

#     # Similarities score
#     for n in tqdm(TRUST_GRAPH):
#         for o_n in TRUST_GRAPH:
#             rated_n = set(RATINGS[n].keys())
#             rated_o_n = set(RATINGS[o_n].keys())
#             rated_common = rated_n.intersection(rated_o_n)
#             union_size = len(rated_n.union(rated_o_n))
            
#             jaccards[n, o_n] = jaccard_index_neighbours(TRUST_GRAPH, n, o_n)
            
#             if len(rated_common) == 0:
#                 item_diffs[n- 1, o_n -1] = 0
#                 item_jaccards[n - 1, o_n -1] = 0
#             else:
#                 diff_sum = 0
#                 for m in rated_common:
#                     diff_sum += abs(RATINGS[n][m] - RATINGS[o_n][m])
#                 avg_diff = diff_sum / len(rated_common) / 4  # Normalizing the average difference
#                 jaccard_index = len(rated_common) / union_size
                
#                 item_diffs[n-1, o_n - 1] = np.float16(avg_diff)
#                 item_jaccards[n-1, o_n -1] = np.float16(jaccard_index)
                
#     return item_diffs, item_jaccards
# Imports
import math
from collections import deque
import random
import sys
from copy import deepcopy
from tqdm.notebook import tqdm
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import os

sys.setrecursionlimit(10**6)


def evaluate(recommender, data, TEST_ITEM=386, TEST_PERCENTAGE=0.3):
    def _get_k(items, k):
        # get k and remove adversaries from test items
        return random.sample([i for i in items if i > 0], k=k)

    test_points = _get_k(data['rated_by'][TEST_ITEM], math.ceil(
        len(data['rated_by'][TEST_ITEM]) * TEST_PERCENTAGE))
    test_data = deepcopy(data)

    # Remove items from test set
    for tp in test_points:
        del test_data['ratings'][tp][TEST_ITEM]
        test_data['rated_by'][TEST_ITEM].remove(tp)

    # Compute recommendations
    recommendations = recommender(
        test_data, TEST_ITEM, test_points)

    # Compute accuracy metrics
    correct, incorrect, differenceSq, difference = 0, 0, 0, 0

    for tp in test_points:
        if round(recommendations[tp]) == data['ratings'][tp][TEST_ITEM]:
            correct += 1
        else:
            incorrect += 1

        differenceSq += pow(recommendations[tp] -
                            data['ratings'][tp][TEST_ITEM], 2)
        difference += abs(recommendations[tp] - data['ratings'][tp][TEST_ITEM])

    acc, mse, diff = correct / \
        len(test_points), differenceSq / \
        len(test_points), difference / len(test_points)
    return acc, mse, diff


def remove_non_voters(graph, ratings, rated_by, ITEMS: list[int]):
    keep_nodes = rated_by[ITEMS[0]]
    # for n in list(keep_nodes):
    #     if len(set(ratings[n].keys()).intersection(ITEMS)) < 2:
    #         keep_nodes.remove(n)# Those who have rated atleast 2 of our items

    reduced_graph = {}
    reduced_ratings = {}
    reduced_rated_by = {i: set([]) for i in ITEMS}

    for n in keep_nodes:
        reduced_graph[n] = list(set(graph[n]).intersection(keep_nodes))
        reduced_ratings[n] = {i: ratings[n][i]
                              for i in ratings[n] if i in ITEMS}
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


def DFSUtil(graph, v, visited, curr_SCC):
    # Mark the current node as visited and print it
    visited.add(v)
    curr_SCC.append(v)
    # Recur for all the vertices adjacent to this vertex
    for i in graph[v]:
        if i not in visited:
            DFSUtil(graph, i, visited, curr_SCC)


def fillOrder(graph, v, visited, stack):
    # Mark the current node as visited
    visited.add(v)
    # Recur for all the vertices adjacent to this vertex
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


def item_jaccard_compute(ratings, v1, v2):
    s1 = set(ratings[v1].keys())
    s2 = set(ratings[v2].keys())
    if len(s1) == len(s2) and len(s1) == 0:
        return 0
    jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
    return jaccard


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


def create_false_node_adversary_graph(name: str, data, n_accounts=1000):
    new_data = deepcopy(data)
    fake_accounts = [i for i in range(-2, -n_accounts-2, -1)]

    # Choose adversarial node to be the one with most incomming nodes
    graph_nodes = sorted(
        list(data['trust_graph'].items()), key=lambda x: len(x[1]), reverse=True)

    # Arbitrarily the 10th most popular node
    adversary_node = graph_nodes[10]

    # Add adversary node to the graph and convert to -1 ID
    new_data['trust_graph'][-1] = adversary_node[1]
    new_data['ratings'][-1] = new_data['ratings'][adversary_node[0]]
    del new_data['trust_graph'][adversary_node[0]]
    del new_data['ratings'][adversary_node[0]]

    for node in new_data['trust_graph']:
        if adversary_node[0] in new_data['trust_graph'][node]:
            new_data['trust_graph'][node].remove(adversary_node[0])
            new_data['trust_graph'][node].append(-1)

    for item in new_data['rated_by']:
        if adversary_node[0] in new_data['rated_by'][item]:
            new_data['rated_by'][item].remove(adversary_node[0])
            new_data['rated_by'][item].add(-1)

    # Given data, takes the target item and adds fake positive ratings to the item
    for item in new_data['rated_by']:
        new_data['rated_by'][item].update(fake_accounts)

    # Rate all items with 1
    new_data['ratings'].update(
        {i: {o_item: 1 for o_item in data['rated_by']} for i in fake_accounts})

    # all fake accounts trust the adversary node
    for i in fake_accounts:
        new_data['trust_graph'][i] = [-1]

    # write to new files
    write_to_file(name, new_data)

    return new_data


def write_to_file(name: str, data):
    # Create directories if they don't exist

    if not os.path.exists(f'../datasets/{name}'):
        os.makedirs(f'../datasets/{name}')

    with open(f'../datasets/{name}/ratings_data.txt', 'w') as f:
        for s in data['ratings']:
            for m in data['ratings'][s]:
                f.write(f'{s} {m} {data["ratings"][s][m]}\n')

    with open(f'../datasets/{name}/trust_data.txt', 'w') as f:
        for s in data['trust_graph']:
            for m in data['trust_graph'][s]:
                f.write(f'{s} {m} 1\n')

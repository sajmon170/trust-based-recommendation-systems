
import random
import numpy as np
from utils import *
from common import *
from collections import deque


def jacc_random_walk_mc(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    def _out_voters(v):
        return len(set(data['trust_graph'][v]).intersection(set(recommendations.keys())))

    def _monte_carlo(node):
        def _traverse(n, d=0, alpha=0.05):
            if n in recommendations:
                return recommendations[n]

            # With alpha * d probability skip to a random node
            if random.random() < d * alpha:
                return _traverse(random.choice(list(recommendations.keys())))

            next_nodes = list(set(data['trust_graph'][n]) - {n})
            def softmax(x): return np.exp(x)/sum(np.exp(x))

            next_jaccs = softmax(
                list([data['jaccard_idx'][n][j] for j in next_nodes]))
            if len(next_nodes) == 0:
                return None

            return _traverse(np.random.choice(next_nodes, p=next_jaccs), d + 1)

        results = []
        for _ in range(100):
            cr = _traverse(node)
            if cr is not None:
                results.append(cr)

        if len(results) == 0:
            return None
        return np.mean(results)

    q = deque(sorted(list(test_points),
              key=lambda x: _out_voters(x), reverse=True))

    while (len(q)):
        current = q.popleft()
        res = _monte_carlo(current)
        if res is None:
            res = random.randint(1, 5)
        recommendations[current] = res

    return recommendations


def item_jaccard_random_walk(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}
    all_users = set(data['trust_graph'].keys())

    def _out_voters(v):
        return len(set(data['trust_graph'][v]).intersection(set(recommendations.keys())))

    def _monte_carlo(node):
        def _traverse(n, d=0, alpha=0.05):
            if n in recommendations:
                return recommendations[n]

            # With alpha * d probability skip to a random node
            if random.random() < d * alpha:
                return _traverse(random.choice(list(recommendations.keys())))

            next_nodes = list(all_users - {n})
            def softmax(x): return np.exp(x)/sum(np.exp(x))

            next_jaccs = softmax(
                list([(data['item_jaccard'][n][j] + data['jaccard_idx'][n][j])*np.mean([data['jaccard_item_sims'][item][i] for i in data['ratings'][j]]) for j in next_nodes]))
            if len(next_nodes) == 0:
                return None

            return _traverse(np.random.choice(next_nodes, p=next_jaccs), d + 1)

        results = []
        for _ in range(100):
            cr = _traverse(node)
            if cr is not None:
                results.append(cr)

        if len(results) == 0:
            return None
        return np.mean(results)

    q = deque(sorted(list(test_points),
              key=lambda x: _out_voters(x), reverse=True))

    while (len(q)):
        current = q.popleft()
        res = _monte_carlo(current)
        if res is None:
            res = random.randint(1, 5)
        recommendations[current] = res

    return recommendations

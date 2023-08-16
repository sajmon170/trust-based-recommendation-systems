from collections import deque
import random
import numpy as np
from utils import *
from common import *

def jacc_random_walk_mc(graph, ratings, rated_by, item, test_points):
    rb = set(rated_by[item])
    recommendations = {n : ratings[n][item] for n in rb}
    jacc_graph = jacc_graph = create_jacc_connected_graph(graph, JACCARD_MATRIX)
    
    def _out_voters(v):
        return len(set(graph[v]).intersection(set(recommendations.keys())))
    
    def _monte_carlo(node):
        def _traverse(n, d=0, alpha=0.05):
            if n in recommendations:
                return recommendations[n]
            
            # With alpha * d probability skip to a random node
            if random.random() < d * alpha:
                return _traverse(random.choice(list(recommendations.keys())))
            
            next_nodes = [o_n for o_n, _ in jacc_graph[n]]
            softmax = lambda x: np.exp(x)/sum(np.exp(x))
            next_jaccs = softmax([jacc for _, jacc in jacc_graph[n]])
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
        
    q = deque(sorted(list(test_points), key = lambda x: _out_voters(x), reverse=True))
    
    while (len(q)):
        current = q.popleft()
        res = _monte_carlo(current)
        if res is None:
            res = random.randint(1, 5)
        recommendations[current] = res
        
    return recommendations
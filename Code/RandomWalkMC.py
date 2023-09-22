from collections import deque
import random
import numpy as np
from utils import *


def random_walk_mc(data, item, test_points):
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

            next = data['trust_graph'][n]
            if len(next) == 0:
                return None

            return _traverse(random.choice(next), d + 1)

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

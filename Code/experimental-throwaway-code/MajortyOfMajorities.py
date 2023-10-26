from statistics import multimode
from random import choice
import random
from utils import jaccard_index, JACCARDS, create_jacc_connected_graph
import numpy as np


def majority_of_majorities_jc(graph, ratings, rated_by, item):
    rb = set(rated_by[item])
    recommendations = {n: ratings[n][item] for n in rb}

    def _out_voters(v):
        return len(set(graph[v]).intersection(set(recommendations.keys())))

    # INSTEAD OF DOING THE COMPLEX STUFF JUST SUM THE JACCARD EDGES TO EACH VALUE AND TAKE THE MAX
    def _surrounding_majority(v):
        jaccards = [0 for _ in range(5)]
        for o_v in graph[v]:
            if o_v in recommendations:
                jaccards[recommendations[o_v] -
                         1] += jaccard_index(graph, v, o_v)

        rating = np.argmax(jaccards) + 1
        if np.max(jaccards) == 0:
            return random.randint(1, 5)
        return rating

    q = sorted(list(set(graph.keys()) - rb),
               key=lambda x: _out_voters(x), reverse=True)
    while len(q) != 0:
        current = q.pop(0)
        recommendations[current] = _surrounding_majority(current)

    return recommendations


def majority_of_majorities_2h(graph, ratings, rated_by, item):
    rb = set(rated_by[item])
    recommendations = {n: ratings[n][item] for n in rb}

    def _out_voters(v):
        return len(set(graph[v]).intersection(set(recommendations.keys())))

    def _surrounding_majority(v):
        surrounding_recs = []
        for o_v in graph[v]:
            if o_v in recommendations:
                surrounding_recs.append(recommendations[o_v])
            for oo_v in graph[o_v]:
                if oo_v in recommendations:
                    surrounding_recs.append(recommendations[oo_v])
        if len(surrounding_recs) == 0:
            return random.randint(1, 5)
        return choice(multimode(surrounding_recs))

    q = sorted(list(set(graph.keys()) - rb),
               key=lambda x: _out_voters(x), reverse=True)
    while len(q) != 0:
        current = q.pop(0)
        recommendations[current] = _surrounding_majority(current)

    return recommendations


def jacc_maj(graph, ratings, rated_by, item):
    rb = set(rated_by[item])
    recommendations = {n: ratings[n][item] for n in rb}

    jacc_graph = create_jacc_connected_graph(graph, JACCARDS)

    for node in list(set(jacc_graph.keys()) - rb):
        jacc_vals = [0 for _ in range(5)]
        for o_v, jacc in jacc_graph[node]:
            if o_v in recommendations:
                jacc_vals[recommendations[o_v] - 1] += jacc
        recommendations[node] = np.argmax(jacc_vals) + 1
        if np.max(jacc_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def top_k_maj(k, *args):
    def _inner_func(graph, ratings, rated_by, item):
        rb = set(rated_by[item])
        recommendations = {n: ratings[n][item] for n in rb}

        jaccards = {}
        for u in list(set(graph.keys()) - rb):
            for o_u in rb:
                if u not in jaccards:
                    jaccards[u] = {}
                jacc = utils.jaccard_index(graph, u, o_u)
                jaccards[u][o_u] = jacc

        for node in jaccards:
            sorted_jaccs = sorted(
                jaccards[node].items(), key=lambda x: x[1], reverse=True)[:k]
            recs = [recommendations[o_v] for o_v, _ in sorted_jaccs]
            recommendations[node] = choice(multimode(recs))
        return recommendations

    return _inner_func

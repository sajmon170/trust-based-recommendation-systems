import math
import random
import numpy as np
from common import TOP_100_ITEM_SIMS
from scipy import stats

def phi(current_user, item, k, trust_graph, ratings, rated_by):
    sigmoid_scalar = 1 / (1 + pow(math.e, -k/2))
    m = -1
    best_j = None
    if current_user not in ratings:
        return 1
    for j in ratings[current_user]:
        new_m = TOP_100_ITEM_SIMS[item][j] * sigmoid_scalar
        if new_m > m:
            m = new_m
            best_j = j
    return m, best_j

def random_walk(user, item, trust_graph, ratings, rated_by):
    # Continue our random walk until broken
    current_user = user
    k = 0
    MAX_DEPTH = 6
    while True:
        # If we go deeper than 6, cancel our walk
        if k >= MAX_DEPTH:
            return None
        
        # If the current user already has rated the item, return it
        if current_user in ratings and item in ratings[current_user]:
            return ratings[current_user][item]
        
        # With probability phi we cancel our random walk or if we are at a dead end
        p, closest_item = phi(current_user, item, k, trust_graph, ratings, rated_by)
        if random.random() < p or len(trust_graph[current_user]) == 0:
            if current_user not in ratings:
                return None
            if len(ratings[current_user]) == 0:
                return None
            
            return ratings[current_user][closest_item]
        
        # Otherwise continue our walk
        if current_user in trust_graph:
            current_user = random.choice(trust_graph[current_user])
            k += 1
        else:
            return None
        
def produce_recommendation(user, item, graph, ratings, rated_by):
    random_walks = []
    
    for _ in range(100):
        rw = random_walk(user, item, graph, ratings, rated_by)
        if rw is not None:
            random_walks.append(rw)
    
    if len(random_walks) == 0:
        return random.randint(1, 5)
    return stats.mode(random_walks)[0][0]
        
        
def trust_walk_recommender(graph, ratings, rated_by, item, test_points):
    rb = set(rated_by[item])
    recommendations = {n : ratings[n][item] for n in rb}
    
    for node in list(test_points): # loop through nodes which need recommendation
        recommendations[node] = produce_recommendation(node, item, graph, ratings, rated_by)

    return recommendations

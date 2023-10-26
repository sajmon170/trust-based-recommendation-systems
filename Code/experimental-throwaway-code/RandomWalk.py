from utils import *
import random

def random_walk_recommender(graph, ratings, rated_by, item):
    non_voters = list(set(graph.keys()) - rated_by[item])
    votes = {n: ratings[n][item] for n in rated_by[item]}
    currently_investigating = set()

    def _sub_func(v):
        currently_investigating.add(v)
        if v in votes:
            return votes[v]
        
        if not can_reach_voter(graph, rated_by, v, item):
            return random.randint(1, 5)
        
        s = 0
        divisor = 0
        for w in graph[v]:
            if w not in currently_investigating:
                divisor += 1
                votes[w] = _sub_func(w)
                s += votes[w]
        currently_investigating.remove(v)
        if v in non_voters:
            non_voters.remove(v)
            
        if divisor == 0:
            return random.randint(1, 5)
        return s / divisor # average of neighbouring votes
    
    def _out_voters(v):
        return len(set(graph[v]).intersection(set(votes.keys())))
    
    non_voters.sort(key = lambda x: _out_voters(x), reverse=True)
    
    while len(non_voters):
        v = non_voters.pop()
        votes[v] = _sub_func(v)

    return votes

def random_walk_recommender_intra_item(graph, ratings, rated_by, item, test_points):
    t_points = [t for t in test_points]
    votes = {n: ratings[n][item] for n in rated_by[item]}
    currently_investigating = set()

    def _sub_func(v):
        currently_investigating.add(v)
        if v in votes:
            return votes[v]
        
        if not can_reach_voter(graph, rated_by, v, item):
            return random.randint(1, 5)
        
        s = 0
        divisor = 0
        for w in graph[v]:
            if w not in currently_investigating:
                divisor += 1
                votes[w] = _sub_func(w)
                s += votes[w]
        currently_investigating.remove(v)
        if v in t_points:
            t_points.remove(v)
            
        if divisor == 0:
            return random.randint(1, 5)
        return s / divisor # average of neighbouring votes
    
    def _out_voters(v):
        return len(set(graph[v]).intersection(set(votes.keys())))
    
    t_points.sort(key = lambda x: _out_voters(x), reverse=True)
    
    while len(t_points):
        v = t_points.pop()
        votes[v] = _sub_func(v)

    return votes
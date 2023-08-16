from random import randint, choices
import numpy as np

def random_recommender(graph, ratings, rated_by, item, test_points):
    require_ratings = set(test_points)
    ratings = {n: ratings[n][item] for n in rated_by[item]}
    for r in require_ratings:
        ratings[r] = randint(1, 5)
        
    return ratings

def get_frequencies(ratings):
    rating_values = np.array(list(ratings.values()))
    frequencies = {rating: np.count_nonzero(rating_values == rating) / len(rating_values) for rating in range(1, 6)}
    return frequencies
        
def universal_recommender(graph, ratings, rated_by, item, test_points):
    require_ratings = set(test_points)
    ratings = {n: ratings[n][item] for n in rated_by[item]}
    frequencies = get_frequencies(ratings)
    
    for r in require_ratings:
        ratings[r] = choices(list(frequencies.keys()), weights=list(frequencies.values()), k = 1)[0]
    
    return ratings
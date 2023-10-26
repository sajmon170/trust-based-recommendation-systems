# Imports
import numpy as np
import random
import scipy.stats as stats
from scipy.special import softmax
from utils import *
import math
from baselines import *
import json
from item_diffs import item_diff_similarity, weighted_item_diff_similarity

# Combining item jaccard and item difference


def item_similarity_recommender(alpha=0.1):
    def recommender(data, item, test_points):
        rb = set(data['rated_by'][item])
        recommendations = {n: data['ratings'][n][item] for n in rb}

        for node in test_points:  # loop through nodes which need recommendation
            similarity_vals = [0 for _ in range(5)]
            for o_v in data['trust_graph']:
                if o_v == node:
                    continue
                if o_v in recommendations:
                    similarity_vals[recommendations[o_v] - 1] += alpha*(
                        data['item_jaccard'][node][o_v]) + (1-alpha)*(data['item_jaccard'][node][o_v])

            recommendations[node] = np.argmax(similarity_vals) + 1

            if np.max(similarity_vals) == 0:
                recommendations[node] = random.choice([1, 2, 3, 4, 5])

        return recommendations
    return recommender


def intra_item_WA_pearson(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph'][node]:
            if o_v == node:
                continue
            for o_item, rating in data['ratings'][o_v].items():
                similarity_vals[round(rating) -
                                1] += data['item_sims'][item][o_item]

        similarity_vals = softmax(similarity_vals)
        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def intra_item_WA_jacc(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph'][node]:
            if o_v == node:
                continue
            for o_item, rating in data['ratings'][o_v].items():
                similarity_vals[round(
                    rating) - 1] += data['jaccard_item_sims'][item][o_item]

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_jaccard_recommender(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) -
                                1] += (data['item_jaccard'][node][o_v])

        recommendations[node] = np.argmax(similarity_vals) + 1

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_jaccard_recommender_WA(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) -
                                1] += (data['item_jaccard'][node][o_v])

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_rating_diff_recommender(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[recommendations[o_v] -
                                1] += data['item_jaccard'][node][o_v]

        recommendations[node] = np.argmax(similarity_vals) + 1

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_rating_diff_recommender_new(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                s = 0
                intr = (set(data['ratings'][o_v].keys()).intersection(
                    set(data['ratings'][node].keys()) - {item}))
                for i in intr:
                    s += 1 - (abs(data['ratings'][o_v][i] -
                              data['ratings'][node][i])/4)

                s /= max(len(intr), 1)
                similarity_vals[recommendations[o_v] - 1] += s

        recommendations[node] = np.argmax(similarity_vals) + 1

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def weighted_item_diffs_new(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                s = 0
                intr = (set(data['ratings'][o_v].keys()).intersection(
                    set(data['ratings'][node].keys()) - {item}))
                for i in intr:
                    s += (1-(abs(data['ratings'][o_v][i] - data['ratings']
                          [node][i])/4)) * data['item_sims'][item][i]

                s /= max(len(intr), 1)
                similarity_vals[round(recommendations[o_v] - 1)] += s

        recommendations[node] = round(np.average(
            [x[1] for x in sorted_vals[:5]], weights=[x[0] for x in sorted_vals[:5]]))
        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_weighted_majorities(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) -
                                1] += (data['jaccard_idx'][node][o_v])
        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))
        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_majority_of_majorities(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) -
                                1] += (data['jaccard_idx'][node][o_v])
        recommendations[node] = np.argmax(similarity_vals) + 1
        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_weighted_neighbours(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph'][node]:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) -
                                1] += (data['jaccard_idx'][node][o_v])
        recommendations[node] = np.argmax(similarity_vals) + 1
        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def mode_of_neighbours(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        ratings = []
        for neighbour in data['trust_graph'][node]:
            if neighbour in recommendations:
                ratings.append(recommendations[neighbour])

        if len(ratings) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])
        else:
            recommendations[node] = stats.mode(ratings)[0][0]

    return recommendations


def mean_of_neighbours(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        ratings = []
        for neighbour in data['trust_graph'][node]:
            if neighbour in recommendations:
                ratings.append(recommendations[neighbour])

        if len(ratings) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])
        else:
            recommendations[node] = np.mean(ratings)

    return recommendations


def median_of_neighbours(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):  # loop through nodes which need recommendation
        ratings = []
        for neighbour in data['trust_graph'][node]:
            if neighbour in recommendations:
                ratings.append(recommendations[neighbour])

        if len(ratings) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])
        else:
            recommendations[node] = np.median(ratings)

    return recommendations


def jaccard_intra_item_jii(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            for o_item, rating in data['ratings'][o_v].items():
                similarity_vals[round(
                    rating) - 1] += data['jaccard_idx'][node][o_v]*(data['jaccard_item_sims'][item][o_item])

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_item_jaccard(alpha=0.1):
    def recommender(data, item, test_points):
        rb = set(data['rated_by'][item])
        recommendations = {n: data['ratings'][n][item] for n in rb}

        for node in test_points:  # loop through nodes which need recommendation
            similarity_vals = [0 for _ in range(5)]
            for o_v in data['trust_graph']:
                if o_v == node:
                    continue
                if o_v in recommendations:
                    similarity_vals[round(recommendations[o_v]) - 1] += alpha*(
                        data['jaccard_idx'][node][o_v]) + (1-alpha)*(data['item_jaccard'][node][o_v])

            recommendations[node] = np.sum(np.array(
                [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

            if np.max(similarity_vals) == 0:
                recommendations[node] = random.choice([1, 2, 3, 4, 5])

        return recommendations
    return recommender


def jaccard_item_jaccard_JII(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                sim_rating = 0.5*(data['jaccard_idx'][node][o_v]) + \
                    0.5*(data['item_jaccard'][node][o_v])
                for o_item, rating in data['ratings'][o_v].items():
                    similarity_vals[round(rating) - 1] += sim_rating*(
                        data['jaccard_item_sims'][item][o_item]) / len(data['ratings'][o_v])

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_item_jaccard_max(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) - 1] += max(
                    data['jaccard_idx'][node][o_v], data['item_jaccard'][node][o_v])

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_item_jaccard_mult(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(
                    recommendations[o_v]) - 1] += data['jaccard_idx'][node][o_v]*data['item_jaccard'][node][o_v]

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_item_jaccard_power_scaling(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in test_points:  # loop through nodes which need recommendation
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) - 1] += (0.5*math.sqrt(
                    data['jaccard_idx'][node][o_v])) + 0.5*data['item_jaccard'][node][o_v]

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_item_rating_item_sim_add(alpha=0.1):
    def recommender(data, item, test_points):
        rb = set(data['rated_by'][item])
        recommendations = {n: data['ratings'][n][item] for n in rb}

        for node in test_points:  # loop through nodes which need recommendation
            similarity_vals = [0 for _ in range(5)]
            for o_v in data['trust_graph']:
                if o_v == node:
                    continue
                if o_v in recommendations:
                    similarity_vals[round(recommendations[o_v]) - 1] += alpha*(item_diff_similarity(
                        node, o_v, data, item)) + (1-alpha)*(data['item_jaccard'][node][o_v])

            recommendations[node] = np.sum(np.array(
                [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

            if np.max(similarity_vals) == 0:
                recommendations[node] = random.choice([1, 2, 3, 4, 5])

        return recommendations
    return recommender


def jaccard_item_rating_power_scaling(alpha=1):
    def recommender(data, item, test_points):
        rb = set(data['rated_by'][item])
        recommendations = {n: data['ratings'][n][item] for n in rb}

        for node in test_points:  # loop through nodes which need recommendation
            similarity_vals = [0 for _ in range(5)]
            for o_v in data['trust_graph']:
                if o_v == node:
                    continue
                if o_v in recommendations:
                    similarity_vals[recommendations[o_v] - 1] += max(
                        data['jaccard_idx'][node][o_v]**(1/alpha), data['item_jaccard'][node][o_v])

            recommendations[node] = np.argmax(similarity_vals) + 1

            if np.max(similarity_vals) == 0:
                recommendations[node] = random.choice([1, 2, 3, 4, 5])

        return recommendations
    return recommender


def weighted_item_diffs(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v] - 1)] += weighted_item_diff_similarity(
                    node, o_v, data, data['item_sims'], item)

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jii_weighted_item_diffs(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v] - 1)] += weighted_item_diff_similarity(
                    node, o_v, data, data['jaccard_item_sims'], item)

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def jaccard_weighted_item_diffs(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(recommendations[o_v]) - 1] += data['jaccard_idx'][node][o_v] + \
                    weighted_item_diff_similarity(
                        node, o_v, data, data['jaccard_item_sims'], item)

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_diff_recommender(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[round(
                    recommendations[o_v]) - 1] += item_diff_similarity(node, o_v, data, item)

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations


def item_diff_recommender_WA(data, item, test_points):
    rb = set(data['rated_by'][item])
    recommendations = {n: data['ratings'][n][item] for n in rb}

    for node in list(test_points):
        similarity_vals = [0 for _ in range(5)]
        for o_v in data['trust_graph']:
            if o_v == node:
                continue
            if o_v in recommendations:
                similarity_vals[min(max(round(recommendations[o_v] - 1), 0), 4)
                                ] += item_diff_similarity(node, o_v, data, item)

        recommendations[node] = np.sum(np.array(
            [i + 1 for i in range(5)]) * np.array(similarity_vals)/np.sum(similarity_vals))

        if np.max(similarity_vals) == 0:
            recommendations[node] = random.choice([1, 2, 3, 4, 5])

    return recommendations

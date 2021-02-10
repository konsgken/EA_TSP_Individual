import numpy as np
from individual import Individual
import sys
from scipy.spatial import distance

def elimination(offspring: list, population: list, lambda_: int) -> list:
    """ Performs the (λ + μ)-elimination step of the evolutionary algorithm.

    Args:
        offspring (list): List of the offspring.
        population (list): List of the individuals in a population.
        lambda_ (int): Number of top lambda_ candidates that will be retained.

    Returns:
        new_combined: Top lambda_ candidates that retained.

    """
    combined = population + offspring
    new_combined = sorted(combined, key=lambda k: k.distance, reverse=False)
    return new_combined[0:lambda_]


def shared_fitness_wrapper_old(selected, survivors, beta_init=0):
    alpha = 1
    sigma = 50

    modObjv = np.zeros(len(selected))

    for idx, x in enumerate(selected):
        ds = []
        one_plus_beta = beta_init
        for survivor in survivors:
            d = distance.hamming(x.route, survivor.route) * len(survivor.route)
            if d <= sigma:
                one_plus_beta += 1 - (d / sigma) ** alpha
        fval = x.route_distance()
        modObjv[idx] = fval * one_plus_beta ** np.sign(fval)

    return modObjv


def shared_fitness_wrapper(selected, survivors, iteration, modObjv_previous=np.empty(0), beta_init=0):
    alpha = 1
    sigma = 15
    modObjv_new = np.zeros(len(selected))
    if iteration == 0:
        for idx, x in enumerate(selected):
            one_plus_beta = beta_init
            survivor = survivors[-1]
            d = distance.hamming(x.route, survivor.route) * len(survivor.route)
            if d <= sigma:
                one_plus_beta = len(survivors) * (1 - (d / sigma) ** alpha) + 1
            fval = x.route_distance()
            modObjv_new[idx] = fval * one_plus_beta ** np.sign(fval)
    if iteration == 1:
        for idx, x in enumerate(selected):
            one_plus_beta = beta_init
            fval = x.route_distance()
            modObjv_new[idx] = fval * one_plus_beta ** np.sign(fval)
    if iteration >= 2:
        for idx, x in enumerate(selected):
            ds = []
            one_plus_beta = beta_init
            survivor = survivors[-1]
            d = distance.hamming(x.route, survivor.route) * len(survivor.route)
            if d <= sigma:
                one_plus_beta = 1 - (d / sigma) ** alpha
                fval = x.route_distance()
                modObjv_new[idx] = modObjv_previous[idx] + fval * one_plus_beta ** np.sign(fval)
            else:
                modObjv_new[idx] = modObjv_previous[idx]
    return modObjv_new


def shared_elimination(offspring, population, lambda_):

    survivors = [population[0] for _ in range(lambda_)]
    combined = population + offspring
    for i in range(lambda_):
        if i < 2:
            fvals = shared_fitness_wrapper(combined, survivors[0:i - 1], i, beta_init=1)
        if i >= 2:
            fvals = shared_fitness_wrapper(combined, survivors[0:i - 1], i, modObjv_previous=fvals, beta_init=1)
        #fvals_new = shared_fitness_wrapper_old(combined, survivors[0:i-1], beta_init=1)
        idx = np.argmin(fvals)
        survivors[i] = combined[idx]
    return survivors

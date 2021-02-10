""" selection_operators.py - Contains various selection operators for
    evolutionary algorithms as well as a wrapper function for using the
    operators
"""

import numpy as np
from individual import Individual
from scipy.spatial import distance


def selection(population: list, k: int, method: int = 0) -> Individual:
    """ Performs selection and returns an individual from a population

    Args:
        population (list): List of the individuals in a population
        k (int): Number of individuals that engage in the tournament selection

    Returns:
        Individual: The individual that won the tournament selection

    Raises:
        ValueError: In case a method is not implemented yet

    """

    if method == 0:
        ind = k_tournament_selection(population, k)
    elif method == 1:
        ind = k_tournament_selection_with_fitness_sharing(population, k)
    else:
        raise ValueError('Method with id={method} is not implemented yet')

    return ind


def k_tournament_selection(population: list, k: int = 3):
    """ Performs k-tournament selection

    Args:
        population (list): List of the individuals in a population
        k (int): Number of individuals that engage in the tournament selection

    Returns:
        Individual: The individual that won the tournament selection

    """

    # Generate a list of k random indices from the population
    indices = np.random.choice(range(len(population)), k)

    # create a list of individuals based on the selected indices
    selected = [population[idx] for idx in indices]

    # calculate and store in a list the fitness for all k individuals
    fitnesses = [s.route_distance() for s in selected]

    # find the index of the individual with the best fitness value out of the
    # k selected individuals
    min_idx = np.argmin(fitnesses)

    return selected[min_idx]


def k_tournament_selection_with_fitness_sharing(population: list, k: int = 3):
    """ Performs k-tournament selection

    Args:
        population (list): List of the individuals in a population
        k (int): Number of individuals that engage in the tournament selection

    Returns:
        Individual: The individual that won the tournament selection

    """

    # Generate a list of k random indices from the population
    indices = np.random.choice(range(len(population)), k)

    # create a list of individuals based on the selected indices
    selected = [population[idx] for idx in indices]

    # calculate and store in a list the fitness for all k individuals
    fitnesses = shared_fitness_wrapper(selected, population)

    # find the index of the individual with the best fitness value out of the
    # k selected individuals
    min_idx = np.argmin(fitnesses)

    return selected[min_idx]


def shared_fitness_wrapper(selected, population, beta_init=0):
    if population is None:
        return [s.route_distance() for s in selected]

    alpha = 1
    sigma = 1

    modObjv = np.zeros(len(selected))

    for idx, x in enumerate(selected):
        ds = []
        one_plus_beta = beta_init
        for individual in population:
            d = distance.hamming(x.route, individual.route) * len(individual.route)
            if d <= sigma:
                one_plus_beta += 1 - (d / sigma) ** alpha
        fval = x.route_distance()
        modObjv[idx] = fval * one_plus_beta ** np.sign(fval)

    return modObjv

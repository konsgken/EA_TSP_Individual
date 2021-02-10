import numpy as np
from individual import Individual


def swap_mutation(individual: Individual) -> Individual:
    """ Performs swap mutation, where two genes are randomly selected and their values are swapped.

    Args:
        individual (Individual): Individual that will be mutated with a mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    i = np.random.randint(len(individual.route))
    j = np.random.randint(len(individual.route))
    individual.route[i], individual.route[j] = \
        individual.route[j], individual.route[i]
    return individual


def inversion_mutation(individual: Individual) -> Individual:
    """ Performs inversion mutation, where a random sequence of genes is selected
     and the order of the genes in that sequence is reversed.

    Args:
        individual (Individual): Individual that will be mutated with a mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    inversion_indices = np.sort(np.random.choice(len(individual.route), 2))
    individual.route[inversion_indices[0]:inversion_indices[1]] = \
        individual.route[inversion_indices[0]:inversion_indices[1]][::-1]
    return individual


def scramble_mutation(individual: Individual) -> Individual:
    if np.random.rand() < individual.alpha:
        # print("Before scramble mutation: ", individual.route)
        scramble_indices = np.sort(np.random.choice(len(individual.route), 2))
        individual.route[scramble_indices[0]:scramble_indices[1]] = \
            np.random.permutation(individual.route[scramble_indices[0]:scramble_indices[1]])
        # print("After scramble mutation:", individual.route)
    return individual

import Reporter
import numpy as np
import random
from matplotlib import pyplot as plt
import pickle


# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        max_iters = 1000
        lambda_ = 100  # population size
        mu = lambda_//3  # offspring size
        k = 2  # candidates for k-tournament selection
        mutation = inversion_mutation  # or swap_mutation
        convergence_threshold = 1e-5
        best_fitnesses = []
        mean_fitnesses = []
        converged = False
        iters = 0

        # Initialize population
        population = [Individual(distance_matrix) for ii in range(lambda_)]

        while iters < max_iters and not converged:

            # Your code here.
            population = GeneticAlgorithm(
                population,
                mu,
                k,
                mutation)

            fitnesses = [individual.distance for individual in population]
            mean_objective = np.mean(fitnesses)
            best_objective = np.min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
            best_fitnesses.append(best_objective)
            mean_fitnesses.append(mean_objective)

            # increament number of iterations
            iters += 1

            # check for convergence
            converged = iters > 0.1 * max_iters and \
                np.std(best_fitnesses[-int(0.1*max_iters):]) \
                < convergence_threshold

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            time_left = self.reporter.report(
                    mean_objective, best_objective, best_solution)
            if time_left < 0:
                break

            minutes = int(time_left // 60)
            seconds = int(time_left % 60)
            print(f'Iteration #{iters} - Best Fitness={best_objective} - '
                  f'Mean Fitness={mean_objective} - '
                  f'Time left: {minutes:d}\' {seconds}\'\'')

            #  with open('TSP_GA_Experiment.pickle', 'wb') as handle:
            #      pickle.dump({'best_fitnesses': best_fitnesses,
            #                   'mean_fitnesses': mean_fitnesses},
            #                  handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Your code here.
        print('')
        if iters == max_iters:
            print('Reached maximum number of iterations and terminated')
        else:
            print(f'Converged and remaining iterations after #{iters} '
                  'were skipped')
        print(f'Distance of best candidate solution is {best_fitnesses[-1]}')

        return 0


class Individual:
    """ Defines a candidate solution for the TSP problem """

    def __init__(self, distance_matrix):
        self.size = len(distance_matrix)
        self.distance_matrix = distance_matrix
        self.route = np.random.permutation(self.size)
        self.alpha = max(0.01, 0.05 + 0.02*np.random.randn())  # mutation rate
        self.distance = self.route_distance()  # total route distance

    def __getitem__(self, key):
        if key >= self.size:
            raise ValueError('Index out of bounds')

        return self.route[key]

    def __str__(self):
        return f"Route: {self.route}, Total distance: {self.distance}"

    def route_distance(self):
        """ Calculates the fitness as the total distance of the route """
        path_distance = 0

        for idx in range(len(self.route)):
            from_city = self.route[idx]
            to_city = None
            if idx + 1 < len(self.route):
                to_city = self.route[idx + 1]
            else:
                to_city = self.route[0]
            path_distance += self.distance_matrix[from_city, to_city]

        self.distance = path_distance

        return self.distance


def GeneticAlgorithm(population, mu, k, mutation):
    """ Runs an evolutionary algorithm for solving the TSP problem """

    lambda_ = len(population)

    offspring = []
    mutation_probability = 0.1
    prob = np.random.rand()

    for _ in range(mu // 2):
        parents = [selection(population, k, method=0) for _ in range(2)]

        if prob > mutation_probability:
            # perform recombination
            offspring1, offspring2 = crossover(parents[0], parents[1])
        else:
            # perform mutation
            offspring1, offspring2 = [mutation(p) for p in parents]

        # append offsprings
        offspring += [offspring1, offspring2]

    # elimination
    population = elimination(offspring, population, lambda_)

    return population


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


def swap_mutation(individual: Individual) -> Individual:
    """ Performs swap mutation, where two genes are randomly selected and
    their values are swapped.

    Args:
        individual (Individual): Individual that will be mutated with a
        mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    i = np.random.randint(len(individual.route))
    j = np.random.randint(len(individual.route))
    individual.route[i], individual.route[j] = \
        individual.route[j], individual.route[i]
    return individual


def inversion_mutation(individual: Individual) -> Individual:
    """ Performs inversion mutation, where a random sequence of genes is
    selected and the order of the genes in that sequence is reversed.

    Args:
        individual (Individual): Individual that will be mutated with a
        mutation rate alpha.
    Returns:
        individual (Individual): The mutated individual.
    """
    inversion_indices = np.sort(np.random.choice(len(individual.route), 2))
    individual.route[inversion_indices[0]:inversion_indices[1]] = \
        individual.route[inversion_indices[0]:inversion_indices[1]][::-1]
    return individual


def crossover(mother, father):
    size = mother.size

    # combine the mutation probability of the parents
    beta = 2 * np.random.rand() - 0.5
    alpha = mother.alpha + beta * (father.alpha - mother.alpha)
    np.clip(alpha, 0, 1)

    start, end = sorted(np.random.randint(size) for i in range(2))
    child1 = [-1] * size
    child2 = [-1] * size
    child1_inherited = []
    child2_inherited = []
    for i in range(start, end + 1):
        child1[i] = mother.route[i]
        child2[i] = father.route[i]
        child1_inherited.append(mother.route[i])
        child2_inherited.append(father.route[i])

    current_father_position, current_mother_position = 0, 0

    inherited_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        if i in inherited_pos:
            i += 1
            continue

        test_child1 = child1[i]
        if test_child1 == -1:
            father_city = father.route[current_father_position]
            while father_city in child1_inherited:
                current_father_position += 1
                father_city = father.route[current_father_position]
            child1[i] = father_city
            child1_inherited.append(father_city)

        test_child2 = child2[i]
        if test_child2 == -1:  # to be filled
            mother_city = mother.route[current_mother_position]
            while mother_city in child2_inherited:
                current_mother_position += 1
                mother_city = mother.route[current_mother_position]
            child2[i] = mother_city
            child2_inherited.append(mother_city)

        i += 1

    c1 = Individual(mother.distance_matrix)
    c1.route = child1
    c1.distance = c1.route_distance()
    c1.alpha = alpha

    c2 = Individual(mother.distance_matrix)
    c2.route = child2
    c2.distance = c2.route_distance()
    c2.alpha = alpha

    return c1, c2


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
    max_idx = np.argmax(fitnesses)

    return selected[max_idx]

import numpy as np
import Reporter
from scipy.spatial import distance
import time


class r0823033:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distance_matrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        # Initialize parameters
        max_iters = 100
        lambda_ = 100  # population size
        mu = lambda_ // 3  # offspring size
        k = 4  # candidates for k-tournament selection
        mutation = scramble_mutation
        convergence_threshold = 1e-5
        best_fitnesses = []
        mean_fitnesses = []
        converged = False
        # Initialize population
        population = [Individual(distance_matrix) for ii in range(lambda_ - 20)]
        optimized_population = [Individual(distance_matrix, nearest_neighbour=True) for ii in range(20)]
        # optimized_population_2 = [two_opt(Individual(distance_matrix), 0.1) for ii in range(10)]
        population = population + optimized_population

        iters = 0
        while iters < max_iters and not converged:

            # Your code here.
            population = GeneticAlgorithm(
                population,
                mu,
                k,
                mutation,
                iters)

            fitnesses = [individual.route_distance() for individual in population]
            mean_objective = np.mean(fitnesses)
            best_objective = np.min(fitnesses)
            best_solution = population[np.argmin(fitnesses)]
            best_fitnesses.append(best_objective)
            mean_fitnesses.append(mean_objective)

            # increment number of iterations
            iters += 1

            # check for convergence
            converged = iters > 0.1 * max_iters and \
                        np.std(mean_fitnesses[-int(0.1 * max_iters):]) \
                        < convergence_threshold

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best
            #    solution with city numbering starting from 0
            time_left = self.reporter.report(
                mean_objective, best_objective, best_solution.route)
            if time_left < 0:
                print('Out of time')
                break

            minutes = int(time_left // 60)
            seconds = int(time_left % 60)
            print(f'Iteration #{iters} - Best Fitness={best_objective} - '
                  f'Mean Fitness={mean_objective} - '
                  f'Time left: {minutes:d}\' {seconds}\'\'')

        # with open('TSP_GA_Experiment.pickle', 'wb') as handle:
        #     pickle.dump({'best_fitnesses': best_fitnesses,
        #                  'mean_fitnesses': mean_fitnesses},
        #                 handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Your code here.
        print('')
        if iters == max_iters:
            print('Reached maximum number of iterations and terminated')
        else:
            print(f'Converged and remaining iterations after #{iters} '
                  'were skipped')
        print(f'Distance of best candidate solution is {best_fitnesses[-1]}')

        return 0


def GeneticAlgorithm(
        population,
        mu,
        k,
        mutation,
        iters):
    """ Runs an evolutionary algorithm for solving the TSP problem """

    lambda_ = len(population)

    offspring = []
    mutation_probability = 0.1

    prob = np.random.rand()

    for _ in range(mu // 2):

        mutation_probability = min(mutation_probability * prob + 0.05, 0.15)
        prob_two_opt = np.random.rand()
        parents = [selection(population, k, method=0) for _ in range(2)]

        if prob > mutation_probability:
            # perform recombination
            offspring1, offspring2 = crossover(parents[0], parents[1])

        else:
            # perform mutation
            offspring1, offspring2 = [mutation(p) for p in parents]

        # append offsprings
        if prob_two_opt < 0.3 and 4 <= iters <= 20:

            if prob > 0.5:
                offspring += [two_opt_optimized(offspring1), offspring2]
            else:
                offspring += [offspring1, two_opt_optimized(offspring2)]
        elif iters > 20:

            if prob > 0.5:
                offspring += [two_opt_optimized(offspring1), offspring2]
            else:
                offspring += [offspring1, two_opt_optimized(offspring2)]

        else:
            offspring += [offspring1, offspring2]
    # elimination
    population = shared_elimination(offspring, population, lambda_)

    return population


#####################################################################################################
# Implementation of Individual class.
#####################################################################################################


class Individual:

    def __init__(self, distanceMatrix, nearest_neighbour=False):
        self.size = len(distanceMatrix)
        self.distanceMatrix = distanceMatrix
        if nearest_neighbour is True:
            self.route = self.nearest_neighbor(self.size, np.random.randint(self.size), distanceMatrix)
        else:
            self.route = np.random.permutation(self.size)

    def __getitem__(self, key):
        if key >= self.size:
            raise ValueError('Index out of bounds')

        return self.route[key]

    def __str__(self):
        return "Route: {self.route}, Total distance: {self.distance}".format(self=self)

    def route_distance(self):
        """
        Calculates the fitness value of the individual, which is the total route distance

        """
        path_distance = 0

        for ii in range(len(self.route)):
            from_city = self.route[ii]
            to_city = None
            if ii + 1 < len(self.route):
                to_city = self.route[ii + 1]
            else:
                to_city = self.route[0]
            path_distance += self.distanceMatrix[from_city, to_city]

        self.distance = path_distance

        return self.distance

    def nearest(self, last, unvisited, D):

        """
        Return the index of the node which is closest to last."""
        near = unvisited[0]
        min_dist = D[last, near]
        for i in unvisited[1:]:
            if D[last, i] < min_dist:
                near = i
                min_dist = D[last, near]
        return near

    def nearest_neighbor(self, n, i, D):
        """
        Return tour starting from city 'i', using the Nearest Neighbor.

        Uses the Nearest Neighbor heuristic to construct a solution:
        - start visiting city i
        - while there are unvisited cities, follow to the closest one
        - return to city i
        """
        unvisited = [k for k in range(n) if k != i]
        # unvisited = range(n)
        # unvisited.remove(i)
        last = i
        tour = [i]
        while unvisited != []:
            next = self.nearest(last, unvisited, D)
            tour.append(next)
            unvisited.remove(next)
            last = next
        return np.array(tour)


#####################################################################################################
# Implementation of scramble mutation.
#####################################################################################################

def scramble_mutation(individual: Individual) -> Individual:
    scramble_indices = np.sort(np.random.choice(len(individual.route), 2))
    individual.route[scramble_indices[0]:scramble_indices[1]] = \
        np.random.permutation(individual.route[scramble_indices[0]:scramble_indices[1]])
    # print("After scramble mutation:", individual.route)
    return individual


#####################################################################################################
# Implementation of k-tournament selection.
#####################################################################################################

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
    min_idx = np.argmin(fitnesses)

    return selected[min_idx]


#####################################################################################################
# Implementation of Ordered crossover.
#####################################################################################################

def crossover(mother, father):
    size = mother.size

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

    c1 = Individual(mother.distanceMatrix)
    c1.route = np.array(child1)
    # c1.distance = c1.route_distance()

    c2 = Individual(mother.distanceMatrix)
    c2.route = np.array(child2)
    # c2.distance = c2.route_distance()

    return c1, c2


#####################################################################################################
# Implementation of elimination functions.
#####################################################################################################

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
    new_combined = sorted(combined, key=lambda k: k.route_distance(), reverse=False)
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
            if (fval == 0 or np.isinf(fval)) and np.isinf(fval):  # Two lines for the infinity values of tour100
                one_plus_beta = 0.1
            modObjv_new[idx] = fval * one_plus_beta ** np.sign(fval)
    if iteration == 1:
        for idx, x in enumerate(selected):
            one_plus_beta = beta_init
            fval = x.route_distance()
            if (fval == 0 or np.isinf(fval)) and np.isinf(fval):  # Two lines for the infinity values of tour100
                one_plus_beta = 0.1
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
                if (fval == 0 or np.isinf(fval)) and np.isinf(fval):  # Two lines for the inifinity values of tour100
                    one_plus_beta = 0.1
                modObjv_new[idx] = modObjv_previous[idx] + fval * one_plus_beta ** np.sign(fval)
            else:
                modObjv_new[idx] = modObjv_previous[idx]
    return modObjv_new


def shared_elimination(offspring, population, lambda_):
    survivors = [population[10] for _ in range(lambda_)]
    combined = population + offspring
    for i in range(lambda_):
        if i < 2:
            fvals = shared_fitness_wrapper(combined, survivors[0:i - 1], i, beta_init=1)
        if i >= 2:
            fvals = shared_fitness_wrapper(combined, survivors[0:i - 1], i, modObjv_previous=fvals, beta_init=1)
        idx = np.argmin(fvals)
        survivors[i] = combined[idx]
    return survivors


#####################################################################################################
# Implementation of k-opt.
#####################################################################################################

def cost_change(cost_mat, n1, n2, n3, n4):
    return float(cost_mat[n1][n3]) + float(cost_mat[n2][n4]) - float(cost_mat[n1][n2]) - float(cost_mat[n3][n4])


def two_opt_optimized(individual):
    cost_mat = individual.distanceMatrix
    best = individual.route
    start_time = time.time()
    improved = True
    while improved:

        improved = False
        for i in range(1, len(individual.route) - 2):
            for j in range(i + 1, len(individual.route)):
                if j - i == 1:
                    continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        individual.route = best
        individual.distance = individual.route_distance()
        end_time = time.time()
        if end_time - start_time > 10:
            break
    return individual

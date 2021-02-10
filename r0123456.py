import random
import numpy as np
from matplotlib import pyplot as plt
import pickle

import Reporter
from individual import Individual
from mutation_operators import swap_mutation, inversion_mutation, scramble_mutation
from selection_operators import selection
from recombination_operators import crossover
from elimination_operators import elimination, shared_elimination
from plotting import plot_best_fitnesses, plot_fitness_histogram, plot_combined
from k_opt import two_opt, two_opt_optimized


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
        # Initialize parameters
        max_iters = 1000
        lambda_ = 100  # population size
        mu = lambda_ // 4  # offspring size
        k = 2  # candidates for k-tournament selection
        #  mutation = swap_mutation
        mutation = scramble_mutation
        mutate_whole_population = False
        convergence_threshold = 1e-5
        best_fitnesses = []
        mean_fitnesses = []
        converged = False

        # enables/disables plotting
        plot = False

        # Initialize population
        population = [Individual(distance_matrix) for ii in range(lambda_)]
        optimized_population = [Individual(distance_matrix, nearest_neighbour=True) for ii in range(10)]
        #optimized_population_2 = [two_opt(Individual(distance_matrix), 0.1) for ii in range(10)]
        population = population + optimized_population

        iters = 0
        while iters < max_iters and not converged:

            # Your code here.
            population = GeneticAlgorithm(
                population,
                mu,
                k,
                mutation,
                iters,
                mutate_whole_population)

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
                        np.std(best_fitnesses[-int(0.1 * max_iters):]) \
                        < convergence_threshold

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best
            #    solution with city numbering starting from 0
            time_left = self.reporter.report(
                mean_objective, best_objective, best_solution)
            if time_left < 0:
                print('Out of time')
                break

            minutes = int(time_left // 60)
            seconds = int(time_left % 60)
            print(f'Iteration #{iters} - Best Fitness={best_objective} - '
                  f'Mean Fitness={mean_objective} - '
                  f'Time left: {minutes:d}\' {seconds}\'\'')

            if plot:
                #  plot_fitness_histogram(fitnesses)
                plot_combined(fitnesses, best_fitnesses, max_iters)

        with open('TSP_GA_Experiment.pickle', 'wb') as handle:
            pickle.dump({'best_fitnesses': best_fitnesses,
                         'mean_fitnesses': mean_fitnesses},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Your code here.
        print('')
        if iters == max_iters:
            print('Reached maximum number of iterations and terminated')
        else:
            print(f'Converged and remaining iterations after #{iters} '
                  'were skipped')
        print(f'Distance of best candidate solution is {best_fitnesses[-1]}')

        if plot:
            plt.show()

        return 0


def GeneticAlgorithm(
        population,
        mu,
        k,
        mutation,
        iters,
        mutate_whole_population=False
):
    """ Runs an evolutionary algorithm for solving the TSP problem """

    lambda_ = len(population)

    offspring = []
    mutation_probability = 0.1

    prob = np.random.rand()

    for _ in range(mu // 2):
        prob_two_opt = np.random.rand()
        parents = [selection(population, k, method=0) for _ in range(2)]

        if prob > mutation_probability:
            # perform recombination
            offspring1, offspring2 = crossover(parents[0], parents[1])

        else:
            # perform mutation
            offspring1, offspring2 = [mutation(p) for p in parents]

        # append offsprings
        if prob_two_opt < 0.4 and 4 <= iters <= 20:
            offspring += [two_opt_optimized(offspring1), offspring2]
        elif iters > 20:
            offspring += [two_opt_optimized(offspring1), offspring2]
        else:
            offspring += [offspring1, offspring2]
    # mutate the rest of the population if the corresponding parameter is true
    if mutate_whole_population:
        population = [mutation(individual) for individual in population[:lambda_]]

    # elimination
    population = shared_elimination(offspring, population, lambda_)

    return population

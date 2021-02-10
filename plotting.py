import numpy as np
from matplotlib import pyplot as plt


def plot_fitness_histogram(fitnesses: list) -> None:
    """ Plots a graph of the best fitnesses for each iteration

    Args:
        fitnesses (list): The fitnesses that will form the histogram
    """
    fig = plt.gcf()
    fig.clear()
    ax = fig.gca()
    ax.hist(fitnesses, bins=list(np.arange(0, 1e5, 1e4)))
    plt.xlabel('fitness')
    plt.ylabel('Occurencies')
    plt.grid()
    plt.pause(0.05)


def plot_best_fitnesses(best_fitnesses: list) -> None:
    """ Plots a graph of the best fitnesses for each iteration

    Args:
        best_fitnesses (list): The best fitnesses for each iteration to plot
    """
    plt.figure(2)
    plt.plot(best_fitnesses)
    plt.xlabel('Generation')
    plt.ylabel('Fitness of best candidate')
    plt.grid()


def plot_combined(fitnesses: list, best_fitnesses: list,
                  max_iterations: int) -> None:
    """ Plots both the fitnesses of the current iteration and the best fitness
        of each iteration

    Args:
        best_fitnesses (list): The best fitnesses for each iteration to plot
        fitnesses (list): The fitnesses that will form the histogram
        max_iterations (int): Determines the x limit of the graph
    """
    fig = plt.gcf()
    fig.clear()
    plt.subplot(2, 1, 1)
    ax = fig.gca()
    ax.hist(fitnesses, bins=100, range=[0, 1e5])
    ax.set_title('Histogram of distances')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Occurencies')
    ax.set_ylim([0, len(fitnesses)])
    ax.grid(True)

    plt.subplot(2, 1, 2)
    ax = fig.gca()
    ax.plot(best_fitnesses)
    ax.set_title('Graph of Best Candidate Solutions per generation')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Distance')
    ax.set_xlim([0, max_iterations])
    ax.set_ylim([0, 1e5])
    ax.grid()

    plt.pause(0.001)


def plot_monte_carlo_method(results: dict, iteration: int) -> None:
    best_fitnesses = results["best_fitnesses"]
    mean_fitnesses = results["mean_fitnesses"]

    if not "Monte Carlo Plot" in plt.get_figlabels():
        plt.close('all')
        fig = plt.figure("Monte Carlo Plot")
        fig.add_subplot(1, 2, 1)
        fig.add_subplot(1, 2, 2)
        ax = fig.axes
        ax[0].set_xlabel("No. Iteration")
        ax[0].set_ylabel("Minimum fitness value")
        ax[1].set_xlabel("No. Iteration")
        ax[1].set_ylabel("Mean fitness value")
        ax[0].grid(True)
        ax[1].grid(True)

    fig = plt.figure("Monte Carlo Plot")
    ax = fig.axes
    ax[0].plot(np.arange(len(best_fitnesses)), best_fitnesses)
    ax[1].plot(np.arange(len(mean_fitnesses)), mean_fitnesses,
               label="Iteration: " + str(iteration))
    fig.suptitle(" Monte carlo simulation - Evolutionary algorithm for TSP")
    ax[1].legend(bbox_to_anchor=(1.01, 1))

    plt.pause(0.01)

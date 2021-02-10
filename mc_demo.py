#!/usr/bin/env python3

from r0823033 import r0823033
import pickle
from matplotlib import pyplot as plt
from plotting import plot_monte_carlo_method


if __name__ == "__main__":
    for ii in range(1):
        r0823033().optimize("tour929.csv")
        with open("TSP_GA_Experiment.pickle", 'rb') as handle:
            results = pickle.load(handle)
        plot_monte_carlo_method(results, ii)
    plt.show()

#!/usr/bin/env python3

import pickle

from r0123456 import r0123456
#  from r0123456_single_file import r0123456
from plotting import plot_monte_carlo_method
from matplotlib import pyplot as plt

if __name__ == "__main__":
    r0123456().optimize("tour29.csv")
    #  r0123456().optimize("tour100.csv")
    #  r0123456().optimize("tour194.csv")
    #  r0123456().optimize("tour929.csv")

    with open("TSP_GA_Experiment.pickle", 'rb') as handle:
        results = pickle.load(handle)

    plot_monte_carlo_method(results, 0)
    plt.show()

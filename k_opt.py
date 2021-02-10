import numpy as np

# Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
#path_distance = lambda route, distance_matrix: route_distance(route, distance_matrix)
# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r, i, k: np.concatenate((r[0:i], r[k:-len(r) + i - 1:-1], r[k + 1:len(r)]))


def path_distance(route, distanceMatrix):
    """
    Calculates the fitness value of the individual, which is the total route distance

    """
    p = 0

    for ii in range(len(route)):
        from_city = route[ii]
        to_city = None
        if ii + 1 < len(route):
            to_city = route[ii + 1]
        else:
            to_city = route[0]
        p += distanceMatrix[from_city, to_city]

    distance = p

    return distance


def two_opt(individual, improvement_threshold):  # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt

    # route = np.arange(cities.shape[0])  # Make an array of row numbers corresponding to cities.
    improvement_factor = 1  # Initialize the improvement factor.
    best_distance = path_distance(individual.route, individual.distanceMatrix)  # Calculate the distance of the initial path.
    while improvement_factor > improvement_threshold:  # If the route is still improving, keep going!
        distance_to_beat = best_distance  # Record the distance at the beginning of the loop.
        for swap_first in range(1, len(individual.route) - 2):  # From each city except the first and last,
            for swap_last in range(swap_first + 1, len(individual.route)):  # to each of the cities following,
                new_route = two_opt_swap(individual.route, swap_first, swap_last)  # try reversing the order of these cities
                new_route = new_route.astype(int)
                new_distance = path_distance(new_route, individual.distanceMatrix)  # and check the total distance with this modification.
                if new_distance < best_distance:  # If the path distance is an improvement,
                    individual.route = new_route  # make this the accepted best route
                    best_distance = new_distance  # and update the distance corresponding to this route.
        improvement_factor = 1 - best_distance / distance_to_beat  # Calculate how much the route has improved.
    individual.distance = individual.route_distance()
    return individual  # When the route is no longer improving substantially, stop searching and return the route.


def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

def two_opt_optimized(individual):
    cost_mat = individual.distanceMatrix
    best = individual.route
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
    return individual


# if __name__ == "__main__":
#     file = open("tour29.csv")
#     distance_matrix = np.loadtxt(file, delimiter=",")
#     file.close()
#     route = np.random.permutation(len(distance_matrix))
#     k_opt_route = two_opt(route, distance_matrix, 0.001)
#     initial_route_distance = route_distance(route, distance_matrix)
#     k_opt_route_distance = route_distance(k_opt_route, distance_matrix)
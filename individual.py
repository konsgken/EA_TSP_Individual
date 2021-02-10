import numpy as np


class Individual:

    def __init__(self, distanceMatrix, nearest_neighbour=False):
        self.size = len(distanceMatrix)
        self.distanceMatrix = distanceMatrix
        if nearest_neighbour is True:
            self.route = self.nearest_neighbor(self.size, np.random.randint(self.size), distanceMatrix)
        else:
            self.route = np.random.permutation(self.size)
        self.alpha = max(0.01, 0.05 + 0.02 * np.random.randn())  # mutation rate
        self.distance = self.route_distance()  # total route distance

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

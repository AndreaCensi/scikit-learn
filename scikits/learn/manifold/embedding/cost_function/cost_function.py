# -*- coding: utf-8 -*-
"""
Non convex cost function for finding a lower-dimension space
where data can be represented.

This module is not meant to be accessed directly, but through helper
functions that lie in the compression module.
"""

from numpy.ctypeslib import ndpointer, load_library
import numpy
import ctypes
import sys

from ..tools import dist2hd
from scikits.learn.externals.optimization.helpers import ForwardFiniteDifferences

class CostFunction(ForwardFiniteDifferences):
    """
    Wrapper with ctypes around the cost function
    """
    def __init__(self, distances, nb_coords = 2, epsilon = 0.0000001, sigma = 1,
      tau = 60, **kwargs):
        """
        Creates the correct cost function with the good arguments
        """
        ForwardFiniteDifferences.__init__(self)
        sortedDistances = distances.flatten()
        sortedDistances.sort()
        sortedDistances = sortedDistances[distances.shape[0]:]

        self._nb_coords = nb_coords
        self._epsilon = epsilon
        self._sigma = (sortedDistances[sigma * len(sortedDistances) // 100])
        self._tau = (sortedDistances[tau * len(sortedDistances) // 100]) ** 2
        self.distances = distances

        del sortedDistances

    def __call__(self, parameters):
        """
        Computes the cost of a parameter
        """
        estimated_coordinates = parameters.reshape(-1, self._nb_coords)
        estimated_distances = dist2hd(estimated_coordinates,
            estimated_coordinates)

        square_diff = (estimated_distances - self.distances)**2

        cost = numpy.sqrt(self._epsilon + square_diff) * \
            numpy.sqrt((self._tau + square_diff) / self._tau) * \
            self.distances / (self._sigma + self.distances)

        cost[numpy.where(self.distances == 0)] = 0

        return numpy.sum(cost)

    def gradient_(self, parameters):
        """
        Computes the gradient of the function
        """
        estimated_coordinates = parameters.reshape(-1, self._nb_coords)
        estimated_distances = dist2hd(estimated_coordinates,
            estimated_coordinates)

        square_diff = (estimated_distances - self.distances)**2

        cost = (estimated_distances - self.distances) / \
            (numpy.sqrt(self._epsilon + square_diff) * estimated_distances) * \
            numpy.sqrt((self._tau + square_diff) / self._tau) * \
            self.distances / (self._sigma + self.distances)

        cost += numpy.sqrt(self._epsilon + square_diff) * \
            (estimated_distances - self.distances) / numpy.sqrt(self._tau) / \
            (numpy.sqrt(self._tau + square_diff) * estimated_distances) * \
            self.distances / (self._sigma + self.distances)

        cost[numpy.where(self.distances == 0)] = 0

        return numpy.sum(cost[:,:,None] * \
            (estimated_coordinates[:, None] - estimated_coordinates[None,:]),
            axis=0).flatten()

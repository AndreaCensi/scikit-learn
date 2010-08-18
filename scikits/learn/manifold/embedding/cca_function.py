# -*- coding: utf-8 -*-

import numpy
import itertools

from .tools import dist2hd
from scikits.learn.externals.optimization.helpers import ForwardFiniteDifferences

class CostFunction(ForwardFiniteDifferences):
    """
    Cost function for the CCA algorithm (doi: 10.1109/72.554199)
    """
    def __init__(self, distances, nb_coords, max_dist, *args, **kwargs):
        """
        Saves the distances to approximate
        Parameters:
          - distances is the matrix distance that will be used
          - max_dist is a percentage indicating what distance to preserve
        """
        ForwardFiniteDifferences.__init__(self)
        self.distances = distances
        self.len = len(self.distances)

        if max_dist < 100:
            sortedDistances = distances.flatten()
            sortedDistances.sort()
            sortedDistances = sortedDistances[distances.shape[0]:]

            self.max_dist = (
                sortedDistances[max_dist * len(sortedDistances) // 100])
        else:
            self.max_dist = 10e20

    def __call__(self, parameters):
        """
        Computes the cost for a parameter
        """
        params = parameters.reshape((self.len, -1))
        d = dist2hd(params, params)
        dist = self.distances < self.max_dist
        d = (d-self.distances)**2 * dist
        return numpy.sum(d)

    def gradient(self, parameters):
        """
        Gradient of this cost function
        """
        params = parameters.reshape((self.len, -1))
        d = dist2hd(params, params)
        dist = d < self.max_dist

        grad = numpy.zeros(params.shape)
        for (g, x, d_a, d_r, d_ok) in itertools.izip(grad, params, d, self.distances, dist):
            temp = (d_a-d_r)/d_a * (x - params).T * d_ok
            temp[numpy.where(numpy.isnan(temp))] = 0
            g[:]= numpy.sum(temp, axis=1)
        return grad.ravel()

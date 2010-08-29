# -*- coding: utf-8 -*-

"""
Simple optimization
"""

import numpy
import numpy.random
import numpy.linalg
import math

from scikits.learn.externals.optimization import line_search, step, optimizer, criterion

class Modifier(object):
    """
    Recenters the points on each axis
    """
    def __init__(self, nb_coords):
        self.nb_coords = nb_coords

    def __call__(self, parameters):
        points = parameters.reshape((-1, self.nb_coords))
        means = numpy.mean(points, axis = 0)
        return (points - means).ravel()

def optimize_cost_function(distances, function, nb_coords = 2,
    ftol = 0.0001, gtol = 0.0001, iterations_max = 10000, **kwargs):
    """
    Computes a new coordinates system that respects the distances between each point
    Parameters :
      - distances is the distances to respect
      - nb_coords is the number of remaining coordinates
    """

    function = function(distances, nb_coords, **kwargs)
    std = numpy.std(numpy.sqrt(distances))/200
    x0 = numpy.random.normal(0., std, distances.shape[0] * nb_coords)

    err = numpy.seterr(invalid='ignore')

    optimi = optimizer.StandardOptimizerModifying(
      function = function,
      step = step.FRPRPConjugateGradientStep(),
      criterion = criterion.criterion(ftol = ftol, gtol = gtol,
          iterations_max = iterations_max),
      x0 = x0,
      line_search = line_search.StrongWolfePowellRule(),
      post_modifier = Modifier(nb_coords))

    optimal = optimi.optimize()
    optimal = optimal.reshape(-1, nb_coords)

    numpy.seterr(**err)

    return optimal, optimi.state

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

class CostFunction(object):
  """
  Wrapper with ctypes around the cost function
  """
  def __init__(self, distances, nb_coords = 2, epsilon = 0.0000001, sigma = 1, tau = 60, **kwargs):
    """
    Creates the correct cost function with the good arguments
    """
    sortedDistances = distances.flatten()
    sortedDistances.sort()
    sortedDistances = sortedDistances[distances.shape[0]:]

    self._nb_coords = nb_coords
    self._epsilon = epsilon
    self._sigma = sigma

    sigma = (sortedDistances[sigma * len(sortedDistances) // 100])
    self._x1 = tau
    tau = (sortedDistances[tau * len(sortedDistances) // 100]) ** 2
    del sortedDistances
    self.grad = None
    self.distances = distances.copy()
    #self._cf = _cost_function.allocate_cost_function(self.distances.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), distances.shape[0], distances.shape[1], nb_coords, epsilon, sigma, tau)

  def __call__(self, parameters):
    """
    Computes the cost of a parameter
    """
    parameters = parameters.copy()
#    return _cost_function.call_cost_function(self._cf, parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(parameters))

  def gradient(self, parameters):
    """
    Computes the gradient of the function
    """
    self.grad = None
    parameters = parameters.copy()
#    _cost_function.gradient_cost_function(self._cf, parameters.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(parameters), ALLOCATOR(self.allocator))
    return self.grad

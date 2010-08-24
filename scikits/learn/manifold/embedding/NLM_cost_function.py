# -*- coding: utf-8 -*-

import numpy
import itertools

from .tools import dist2hd

class CostFunction(object):
  """
  Cost function for the NLM algorithm
  """
  def __init__(self, distances, *args, **kwargs):
    """
    Saves the distances to approximate
    """
    self.distances = distances
    distances.tofile("test")
    self.factor = numpy.sum(distances)
    self.len = len(self.distances)

  def __call__(self, parameters):
    """
    Computes the cost for a parameter
    """
    params = parameters.reshape((self.len, -1))
    d = dist2hd(params, params)
    d = (d-self.distances)**2/self.distances
    d[numpy.where(numpy.isnan(d))] = 0
    d[numpy.where(numpy.isinf(d))] = 0
    return self.factor * numpy.sum(d)

  def gradient(self, parameters):
    """
    Gradient of this cost function
    """
    params = parameters.reshape((self.len, -1))
    d = dist2hd(params, params)

    grad = numpy.zeros(params.shape)
    for (g, x, d_a, d_r) in itertools.izip(grad, params, d, self.distances):
      temp = 2 * (x - params).T * (d_a-d_r)/(d_r*d_a)
      temp[numpy.where(numpy.isnan(temp))] = 0
      g[:]= numpy.sum(temp, axis=1)
    return grad.ravel()

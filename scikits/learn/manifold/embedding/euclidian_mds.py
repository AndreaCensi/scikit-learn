# -*- coding: utf-8 -*-

import numpy
import math

from ...base_estimator import BaseEstimator
from ..mapping import builder as mapping_builder

from .tools import dist2hd
from .dimensionality_reduction import optimize_cost_function
from .NLM import CostFunction as NLM_CostFunction

def reduct(reduction, function, samples, n_coords, **kwargs):
    """
    Data reduction with euclidian distance approximation:
      - reduction is the algorithm to use
      - function is the function to optimize
      - samples is an array with the samples for the compression
      - n_coords is the number of coordinates that must be retained
    """
    distances = dist2hd(samples, samples)
    return reduction(distances, function, n_coords, **kwargs)

def mds(distances, function, n_coords):
    """
    Computes a new set of coordinates based on the distance matrix passed as a
    parameter, in fact it is a classical MDS
    """
    square_distances = -distances ** 2 / 2.
    correlations = square_distances + numpy.mean(
        square_distances) - numpy.mean(square_distances, axis=0) - numpy.mean(
        square_distances, axis=1)[numpy.newaxis].T
    (u, s, vh) = numpy.linalg.svd(correlations)
    return u[:, :n_coords] * numpy.sqrt(s[:n_coords]), {
        'scaling' : u[:n_coords], 'eigen_vectors' : u[:, :n_coords]}

class NLM(BaseEstimator):
    """
    NLM embedding object

    Parameters
    ----------
    n_coords : int
      The dimension of the embedding space

    n_neighbors : int
      The number of K-neighboors to use (optional, default 9) if neigh is not
      given.

    neigh : Neighbors
      A neighboorer (optional). By default, a K-Neighbor research is done.
      If provided, neigh must be a functor class . `neigh_alternate_arguments`
      will be passed to this class constructor.

    neigh_alternate_arguments : dictionary
      Dictionary of arguments that will be passed to the `neigh` constructor

    mapping_kind : object
      The type of mapper to use. Can be:
          * None : no mapping built
          * "Barycenter" (default) : Barycenter mapping
          * a class object : a class that will be instantiated with the
              arguments of this function
          * an instance : an instance that will be fit() and then
              transform()ed

    temp_file : string
      name of a file for caching the distance matrix

    Attributes
    ----------
    embedding_ : array_like
        Embedding of the learning data

    X_ : array_like
        Original data that is embedded

    See Also
    --------


    Notes
    -----

    .. [1]     Data reduction with NonLinear Mapping algorithm
    (JR. J. Sammon. A nonlinear mapping for data structure analysis.
    IEEE Transactions on Computers, C-18(No. 5):401--409, May 1969):


    Examples
    --------
    >>> from scikits.learn.manifold import NLM
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> nlm = NLM(n_coords = 2, mapping_kind = None)
    >>> nlm = nlm.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, mapping_kind = "Barycenter", temp_file=None):
        self.n_coords = n_coords
        self.n_neighbors = n_neighbors if n_neighbors is not None else 9
        self.neigh = neigh
        self.neigh_alternate_arguments = neigh_alternate_arguments
        self.mapping_kind = mapping_kind
        self.temp_file= temp_file

    def fit(self, X):
        """
        Parameters
        ----------
        X : array_like
        The learning dataset

        Returns
        -------
        Self
        """
        self.X_ = numpy.asanyarray(X)
        self.embedding_ = reduct(optimize_cost_function,
            NLM_CostFunction, self.X_, n_coords = self.n_coords)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

    def transform(self, X):
        if self.mapping:
            return self.mapping.transform(X)
        else:
            raise RuntimeError("No mapping was built for this embedding")

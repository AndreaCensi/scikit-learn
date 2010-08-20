# -*- coding: utf-8 -*-

"""
Dimensionality reduction with geodesic distances
"""

import os
import numpy
import numpy.random
import numpy.linalg
import math

#from scikits.optimization import *

from .embedding import Embedding
from ..mapping import builder as mapping_builder

from .tools import create_neighborer
from .distances import numpy_floyd
from .euclidian_mds import mds as euclidian_mds

from .cca_function import CostFunction as CCA_CostFunction
from .cost_function import CostFunction as RobustCostFunction
from .NLM import CostFunction as NLM_CostFunction

from .dimensionality_reduction import optimize_cost_function
from .multiresolution_dimensionality_reduction import \
    optimize_cost_function as multiresolution_optimize_cost_function
from .cca_multiresolution_dimensionality_reduction import \
    optimize_cost_function as cca_multiresolution_optimize_cost_function
from .robust_dimensionality_reduction import optimize_cost_function \
    as robust_dimensionality_optimize_cost_function

def reduct(reduction, function, samples, n_coords, neigh, n_neighbors,
    neigh_alternate_arguments, temp_file, **kwargs):
    """
    Data reduction with geodesic distance approximation

    Parameters
    ----------
    reduction:
      The reduction technique

    samples : matrix
      The points to consider.

    temp_file : string
      name of a file for caching the distance matrix

    n_neighbors : int
      The number of K-neighboors to use (optional, default 9) if neigh is not
      given.

    neigh : Neighbors
      A neighboorer (optional). By default, a K-Neighbor research is done.
      If provided, neigh must be a functor. All parameters passed to this
      function will be passed to its constructor.
    """
    try:
        dists = numpy.load(temp_file)
    except:
        neigh = create_neighborer(samples, neigh, n_neighbors, 
            neigh_alternate_arguments)

        dists = populate_distance_matrix_from_neighbors(samples, neigh)
        numpy_floyd(dists)
        if temp_file:
            numpy.save(temp_file, dists)

    return reduction(dists, function, n_coords, **kwargs)

def populate_distance_matrix_from_neighbors(points, neighborer):
    """
    Creates a matrix with infinite value safe for points that are neighbors
    """
    distances = numpy.zeros((points.shape[0], points.shape[0]),
        dtype = numpy.float)
    distances[:] = numpy.inf
    for indice in xrange(0, len(points)):
        neighbor_list = neighborer.predict(points[indice])[1]
        for element in neighbor_list:
            d = math.sqrt(
                numpy.sum((points[indice] - points[element])**2))
            distances[indice, element] = d
            distances[element, indice] = d

    return distances

class Isomap(Embedding):
    """
    Isomap embedding object

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

    .. [1] Tenenbaum, J. B., de Silva, V. and Langford, J. C.,
           "A Global Geometric Framework for Nonlinear Dimensionality 
           Reduction,"
           Science, 290(5500), pp. 2319-2323, 2000

    Examples
    --------  
    >>> from scikits.learn.manifold import Isomap
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> isomap = Isomap(n_coords = 2, mapping_kind = None, n_neighbors = 3)
    >>> isomap = isomap.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, mapping_kind = "Barycenter",
        temp_file=None):
        Embedding.__init__(self, n_coords, n_neighbors, neigh,
           neigh_alternate_arguments, mapping_kind)
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
        self.embedding_, self.reduced_parameters_ = reduct(euclidian_mds, 
            None, self.X_, n_coords = self.n_coords, neigh = self.neigh,
            n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            temp_file = self.temp_file)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

class CCA(Embedding):
    """
    CCA embedding object

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

    max_dist : float
      percentage of maximum distance to preserve (default is 99%)

    xtol : float
      Tolerance for the parameters

    iterations_max : integer
      Max number of iterations

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

    .. [1] doi: 10.1109/72.554199

    Examples
    --------
    >>> from scikits.learn.manifold import CCA
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> cca = CCA(n_coords = 2, mapping_kind = None, n_neighbors = 3)
    >>> cca = cca.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, mapping_kind = "Barycenter",
        temp_file=None, max_dist = 99, xtol = 0.01, iterations_max = 100):
        Embedding.__init__(self, n_coords, n_neighbors,
           neigh,neigh_alternate_arguments, mapping_kind)
        self.temp_file = temp_file
        self.max_dist = max_dist
        self.xtol = xtol
        self.iterations_max = iterations_max

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
        self.embedding_ = reduct(cca_multiresolution_optimize_cost_function,
            CCA_CostFunction, self.X_, n_coords = self.n_coords,
            neigh = self.neigh, n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            temp_file = self.temp_file, max_dist = self.max_dist,
            xtol = self.xtol, iterations_max = self.iterations_max)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

class RobustEmbedding(Embedding):
    """
    Robust embedding object with geodesic distances approximation

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

    epsilon : float
      small value for keeping the absolute norm derivable

    tau : float
      Distances higher than tau (percentage) will have a smaller weight

    sigma : float
      Distances lower than sigma (percentage) will have a smaller weight

    ftol : float
      Tolerance for the inner cost function

    gtol : float
      Tolerance for the gradient of the cost function

    iterations_max : integer
      Max number of iterations

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

    .. [1] Brucher, M. et al.



    Examples
    --------
    >>> from scikits.learn.manifold import RobustEmbedding
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> robust = RobustEmbedding(n_coords = 2, mapping_kind = None, \
        n_neighbors=3, ftol = 0.01)
    >>> robust = robust.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, epsilon = 0.0000001, tau = 60,
        sigma = 1, ftol = 0.00000001, gtol = 0.00000001, iterations_max = 10000,
        mapping_kind = "Barycenter", temp_file=None):
        Embedding.__init__(self, n_coords, n_neighbors,
           neigh,neigh_alternate_arguments, mapping_kind)
        self.temp_file= temp_file
        self.epsilon = epsilon
        self.tau = tau
        self.sigma = sigma
        self.ftol = ftol
        self.gtol = gtol
        self.iterations_max = iterations_max

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
        self.embedding_, self.reduced_parameters_ = \
            reduct(robust_dimensionality_optimize_cost_function,
            RobustCostFunction, self.X_, n_coords = self.n_coords,
            neigh = self.neigh, n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            temp_file = self.temp_file, epsilon = self.epsilon,
            tau = self.tau, sigma = self.sigma, ftol = self.ftol,
            gtol = self.gtol, iterations_max = self.iterations_max)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

class RobustMultiresolutionEmbedding(Embedding):
    """
    Robust embedding object with multiresolution optimization

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

    epsilon : float
      small value for keeping the absolute norm derivable

    tau : float
      Distances higher than tau (percentage) will have a smaller weight

    sigma : float
      Distances lower than sigma (percentage) will have a smaller weight

    ftol : float
      Tolerance for the inner cost function

    gtol : float
      Tolerance for the gradient of the cost function

    iterations_max : integer
      Max number of iterations

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

    .. [1] Brucher, M. et al.
           "Data reduction with NonLinear Mapping algorithm"
           IEEE Transactions on Computers, C-18(No. 5):401--409, May 1969


    Examples
    --------
    >>> from scikits.learn.manifold import RobustMultiresolutionEmbedding
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> robust = RobustMultiresolutionEmbedding(n_coords = 2, \
        mapping_kind = None, n_neighbors=3, ftol = 0.01)
    >>> robust = robust.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, epsilon = 0.0000001, tau = 60,
        sigma = 1, ftol = 0.00000001, gtol = 0.00000001, iterations_max = 10000,
        mapping_kind = "Barycenter", temp_file=None):
        Embedding.__init__(self, n_coords, n_neighbors,
           neigh,neigh_alternate_arguments, mapping_kind)
        self.temp_file= temp_file
        self.epsilon = epsilon
        self.tau = tau
        self.sigma = sigma
        self.ftol = ftol
        self.gtol = gtol
        self.iterations_max = iterations_max

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
        self.embedding_, self.reduced_parameters_ = \
            reduct(multiresolution_optimize_cost_function,
            RobustCostFunction, self.X_, n_coords = self.n_coords,
            neigh = self.neigh, n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            temp_file = self.temp_file, epsilon = self.epsilon,
            tau = self.tau, sigma = self.sigma, ftol = self.ftol,
            gtol = self.gtol, iterations_max = self.iterations_max)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

class GeodesicNLM(Embedding):
    """
    NLM embedding object with geodesic distances approximation

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

    ftol : float
      Tolerance for the inner cost function

    gtol : float
      Tolerance for the gradient of the cost function

    iterations_max : integer
      Max number of iterations

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

    .. [1] JR. J. Sammon.,
           "Data reduction with NonLinear Mapping algorithm"
           IEEE Transactions on Computers, C-18(No. 5):401--409, May 1969


    Examples
    --------
    >>> from scikits.learn.manifold import GeodesicNLM
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> nlm = GeodesicNLM(n_coords = 2, mapping_kind = None, n_neighbors=3)
    >>> nlm = nlm.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, ftol = 0.00000001, gtol = 0.00000001,
        iterations_max = 10000, mapping_kind = "Barycenter", temp_file=None):
        Embedding.__init__(self, n_coords, n_neighbors, neigh,
           neigh_alternate_arguments, mapping_kind)
        self.temp_file= temp_file
        self.ftol = ftol
        self.gtol = gtol
        self.iterations_max = iterations_max

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
        self.embedding_, self.reduced_parameters_ = reduct(optimize_cost_function,
        NLM_CostFunction, self.X_, n_coords = self.n_coords, neigh = self.neigh,
            n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            temp_file = self.temp_file, ftol = self.ftol,
            gtol = self.gtol, iterations_max = self.iterations_max)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

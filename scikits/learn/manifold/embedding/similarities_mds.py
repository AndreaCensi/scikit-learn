# -*- coding: utf-8 -*-

"""
Dimensionality reduction with similarities
"""

import numpy as np

from ...preprocessing import Scaler

from .base_embedding import BaseEmbedding
from ..mapping import builder as mapping_builder

from .similarities import laplacian_maps, sparse_heat_kernel, \
    normalized_heat_kernel

def centered_normalized(samples):
    """
    Returns a set of samples that are centered and of variance 1

    >>> import numpy
    >>> from  scikits.learn.manifold.embedding.similarities_mds import centered_normalized
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> centered_normalized(samples)
    array([[-1.08012345, -1.08012345, -0.40824829],
           [ 1.08012345, -1.08012345, -0.40824829],
           [-1.08012345,  1.08012345, -0.40824829],
           [ 1.08012345,  1.08012345, -0.40824829],
           [-1.08012345,  0.        , -0.40824829],
           [ 0.        , -1.08012345, -0.40824829],
           [ 1.08012345,  1.08012345,  2.44948974]])
    """
    scaler = Scaler(with_std=True)
    scaler.fit(samples)
    return scaler.transform(samples)

class LaplacianEigenmap(BaseEmbedding):
    """
    Laplacian Eigenmap embedding object

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

    kernel_width : float
      Width of the heat kernel

    mapping_kind : object
      The type of mapper to use. Can be:
          * None : no mapping built
          * "Barycenter" (default) : Barycenter mapping
          * a class object : a class that will be instantiated with the
              arguments of this function
          * an instance : an instance that will be fit() and then
              transform()ed

    Attributes
    ----------
    embedding_ : array_like
        BaseEmbedding of the learning data

    X_ : array_like
        Original data that is embedded

    See Also
    --------


    Notes
    -----

    .. [1]

    Examples
    --------
    >>> from scikits.learn.manifold import LaplacianEigenmap
    >>> import numpy as np
    >>> samples = np.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> laplacian = LaplacianEigenmap(n_coords=2, mapping_kind=None,\
          n_neighbors=3, kernel_width=.5)
    >>> laplacian = laplacian.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors=None, neigh=None,
        neigh_alternate_arguments=None, mapping_kind="Barycenter",
        kernel_width = .5):
        BaseEmbedding.__init__(self, n_coords, n_neighbors,
            neigh,neigh_alternate_arguments, mapping_kind)
        self.kernel_width = kernel_width

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
        self.X_ = np.asanyarray(X)
        self.embedding_ = laplacian_maps(self.X_, n_coords=self.n_coords,
            neigh=self.neigh, n_neighbors=self.n_neighbors,
            neigh_alternate_arguments=self.neigh_alternate_arguments,
            method=sparse_heat_kernel, kernel_width=self.kernel_width)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh=self.neigh, n_neighbors=self.n_neighbors - 1,
            neigh_alternate_arguments=self.neigh_alternate_arguments)
        return self

class DiffusionMap(BaseEmbedding):
    """
    Diffusion Map embedding object

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

    kernel_width : float
      Width of the heat kernel

    mapping_kind : object
      The type of mapper to use. Can be:
          * None : no mapping built
          * "Barycenter" (default) : Barycenter mapping
          * a class object : a class that will be instantiated with the
              arguments of this function
          * an instance : an instance that will be fit() and then
              transform()ed

    Attributes
    ----------
    embedding_ : array_like
        BaseEmbedding of the learning data

    X_ : array_like
        Original data that is embedded

    See Also
    --------


    Notes
    -----

    .. [1]

    Examples
    --------
    >>> from scikits.learn.manifold import DiffusionMap
    >>> import numpy as np
    >>> samples = np.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> diffusion = DiffusionMap(n_coords=2, mapping_kind=None,\
          n_neighbors=3, kernel_width=.5)
    >>> diffusion = diffusion.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors=None, neigh=None,
        neigh_alternate_arguments=None, mapping_kind="Barycenter",
        kernel_width=.5):
        BaseEmbedding.__init__(self, n_coords, n_neighbors,
            neigh,neigh_alternate_arguments, mapping_kind)
        self.kernel_width = kernel_width

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
        self.X_ = np.asanyarray(X)
        self.embedding_ = laplacian_maps(centered_normalized(self.X_),
            n_coords = self.n_coords,
            neigh = self.neigh, n_neighbors = self.n_neighbors,
            neigh_alternate_arguments = self.neigh_alternate_arguments,
            method=normalized_heat_kernel, kernel_width=self.kernel_width)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh=self.neigh, n_neighbors=self.n_neighbors - 1,
            neigh_alternate_arguments=self.neigh_alternate_arguments)
        return self

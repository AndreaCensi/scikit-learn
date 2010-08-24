# -*- coding: utf-8 -*-

"""
PCA module, by Zachary Pincus
"""

import numpy
import numpy.linalg

from .embedding import Embedding
from ..mapping import builder as mapping_builder

class PCA(Embedding):
    """
    PCA embedding object

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

    Examples
    --------
    >>> from scikits.learn.manifold import PCA
    >>> import numpy
    >>> samples = numpy.array((0., 0., 0., \
      1., 0., 0., \
      0., 1., 0., \
      1., 1., 0., \
      0., .5, 0., \
      .5, 0., 0., \
      1., 1., 0.5, \
      )).reshape((-1,3))
    >>> pca = PCA(n_coords = 2, mapping_kind = None)
    >>> pca = pca.fit(samples)
    """
    def __init__(self, n_coords, n_neighbors = None, neigh = None,
        neigh_alternate_arguments = None, mapping_kind = "Barycenter", temp_file=None):
        Embedding.__init__(self, n_coords, n_neighbors,
            neigh,neigh_alternate_arguments, mapping_kind)
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
        self.embedding_, self.reduced_parameters_ = self._project(self.X_, n_coords = self.n_coords)
        self.mapping = mapping_builder(self, self.mapping_kind,
            neigh = self.neigh, n_neighbors = self.n_neighbors - 1,
            neigh_alternate_arguments = self.neigh_alternate_arguments)
        return self

    def _project(self, samples, n_coords):
        """
        Using a function
        """
        centered = samples - numpy.mean(samples, axis=0)
        try:
            corr = numpy.dot(centered.T, centered)
            (w, v) = numpy.linalg.eigh(corr)
            index = numpy.argsort(w)

            unscaled = v[index[-1:-1-n_coords:-1]]
            vectors = unscaled#numpy.sqrt(w[index[-1:-1-n_coords:-1]])[:,numpy.newaxis] * unscaled
            inv = numpy.linalg.inv(numpy.dot(unscaled.T, unscaled.T))
            return numpy.dot(centered, numpy.dot(vectors.T, inv)), {'v': v}

        except:
            corr = numpy.dot(centered, centered.T)
            (w, v) = numpy.linalg.eigh(corr)
            index = numpy.argsort(w)

            unscaled = v[:,index[-1:-1-n_coords:-1]]
            vectors = numpy.dot(unscaled.T, centered)
            vectors = (1/numpy.sqrt(w[index[-1:-1-n_coords:-1]])[:,numpy.newaxis]) * vectors
        return numpy.dot(centered, vectors.T), {'v': v}

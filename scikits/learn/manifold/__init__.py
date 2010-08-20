# -*- coding: utf-8 -*-

"""
Manifold Learning Module
"""

from .embedding.pca import PCA
from .embedding.geodesic_mds import Isomap, CCA, GeodesicNLM, RobustEmbedding, \
    RobustMultiresolutionEmbedding
from .embedding.euclidian_mds import NLM

from .mapping.barycenter import Barycenter

# -*- coding: utf-8 -*-
"""
===================================
Non-Linear dimensionality reduction
===================================

An illustration of non-linear dimensionality reduction
with the manifold module

"""

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

################################################################################
# import some data to play with

# The IRIS dataset
from scikits.learn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

def display_embedding(embedding, title):
    fig = pl.figure()
    ax = Axes3D(fig)#fig.gca(projection='3d')
    for c, i, m, target_name in zip("rgb", [0, 1, 2], ['o', '^', 's'],
        target_names):
        ax.scatter(embedding[y==i,0], embedding[y==i,1], embedding[y==i,2], c=c,
           label=target_name, marker=m)
    ax.legend()
    ax.set_title(title)

################################################################################
# PCA
#print "Computing PCA embedding"
#from scikits.learn.manifold import PCA
#pca = PCA(n_coords=3)
#X_r = pca.fit(X).embedding_

#display_embedding(X_r, 'PCA embedding of IRIS dataset')

#################################################################################
## Isomap
#print "Computing Isomap embedding"
#from scikits.learn.manifold import Isomap
#isomap = Isomap(n_coords=3, n_neighbors=30)
#X_r = isomap.fit(X).embedding_

#display_embedding(X_r, 'ISOMAP embedding of IRIS dataset')

#################################################################################
## CCA
#print "Computing CCA embedding"
#from scikits.learn.manifold import CCA
#cca = CCA(n_coords=3, n_neighbors=30, max_dist=75)
#X_r = cca.fit(X).embedding_

#display_embedding(X_r, 'CCA embedding of IRIS dataset')

#################################################################################
## NLM
#print "Computing NLM embedding"
#from scikits.learn.manifold import NLM
#nlm = NLM(n_coords=3)
#X_r = nlm.fit(X).embedding_

#display_embedding(X_r, 'NonLinear Mapping embedding of IRIS dataset')

#################################################################################
## geodesic NLM
#print "Computing Geodesic NLM embedding"
#from scikits.learn.manifold import GeodesicNLM
#nlm = GeodesicNLM(n_coords=3, n_neighbors=28)
#X_r = nlm.fit(X).embedding_

#display_embedding(X_r, 'Geodesic NonLinear Mapping embedding of IRIS dataset')

################################################################################
# Robust embedding
print "Computing robust embedding"
from scikits.learn.manifold import RobustEmbedding
#embedding = RobustEmbedding(n_coords=3, n_neighbors=28)
#X_r = embedding.fit(X).embedding_

#display_embedding(X_r, 'Robust embedding of IRIS dataset')

################################################################################
# Robust embedding
print "Computing robust multiresolution embedding"
from scikits.learn.manifold import RobustMultiresolutionEmbedding
#embedding = RobustMultiresolutionEmbedding(n_coords=3, n_neighbors=28)
#X_r = embedding.fit(X).embedding_

#display_embedding(X_r,
#    'Robust Multiresolution embedding of IRIS dataset')

################################################################################
# LLE
print "Computing LLE embedding"
from scikits.learn.manifold import LLE
embedding = LLE(n_coords=3, n_neighbors=28)
X_r = embedding.fit(X).embedding_

display_embedding(X_r, 'LLE embedding of IRIS dataset')

################################################################################
# Laplacian Eigenmap
print "Computing Laplacian Eigenmap embedding"
from scikits.learn.manifold import LaplacianEigenmap
embedding = LaplacianEigenmap(n_coords=3, n_neighbors=30)
X_r = embedding.fit(X).embedding_
print X_r
print y
display_embedding(X_r, 'Laplacian Eigenmap embedding of IRIS dataset')

################################################################################
# Diffuion Map
print "Computing Diffusion Map embedding"
from scikits.learn.manifold import DiffusionMap
embedding = DiffusionMap(n_coords=3, kernel_width=10)
X_r = embedding.fit(X).embedding_

display_embedding(X_r, 'Diffusion Map embedding of IRIS dataset')

pl.show()

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

################################################################################
# Isomap
from scikits.learn.manifold import Isomap
isomap = Isomap(n_coords=3, n_neighbors=30)
X_r = isomap.fit(X).embedding_

fig = pl.figure()
ax = Axes3D(fig)#fig.gca(projection='3d')
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    ax.scatter(X_r[y==i,0], X_r[y==i,1], X_r[y==i,2], c=c, label=target_name)
ax.legend()
ax.set_title('ISOMAP embedding of IRIS dataset')

################################################################################
# NLM
from scikits.learn.manifold import NLM
nlm = NLM(n_coords=3)
X_r = nlm.fit(X).embedding_

fig = pl.figure()
ax = Axes3D(fig)#fig.gca(projection='3d')
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    ax.scatter(X_r[y==i,0], X_r[y==i,1], X_r[y==i,2], c=c, label=target_name)
ax.legend()
ax.set_title('NonLinear Mapping embedding of IRIS dataset')

################################################################################
# NLM
from scikits.learn.manifold import GeodesicNLM
nlm = GeodesicNLM(n_coords=3, n_neighbors=28)
X_r = nlm.fit(X).embedding_

fig = pl.figure()
ax = Axes3D(fig)#fig.gca(projection='3d')
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    ax.scatter(X_r[y==i,0], X_r[y==i,1], X_r[y==i,2], c=c, label=target_name)
ax.legend()
ax.set_title('Geodesic NonLinear Mapping embedding of IRIS dataset')

pl.show()

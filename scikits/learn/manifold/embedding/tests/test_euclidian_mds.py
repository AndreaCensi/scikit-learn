#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from numpy.testing import assert_array_equal, \
                          assert_array_almost_equal, \
                          assert_raises
import numpy.random

from unittest import TestCase
from tempfile import NamedTemporaryFile
from nose.tools import raises

from ..tools import create_neighborer, dist2hd
from ..euclidian_mds import NLM

samples = numpy.array((0., 0., 0.,
  1., 0., 0.,
  0., 1., 0.,
  1., 1., 0.,
  0., .5, 0.,
  .5, 0., 0.,
  1., 1., 0.5,
  )).reshape((-1,3))

distances = numpy.array(((0, 1, 1, 2, .5, .5, 2.11803399),
         (1, 0, 2, 3, 1.5, .5, 3.11803399),
         (1, 2, 0, 1, .5, 1.5, 1.11803399),
         (2, 3, 1, 0, 1.5, 2.5, .5),
         (.5, 1.5, .5, 1.5, 0, 1, 1.61803399),
         (.5, .5, 1.5, 2.5, 1, 0, 2.61803399),
         (2.11803399, 3.11803399, 1.11803399, .5, 1.61803399, 2.61803399, 0)))

class TestNLM(TestCase):
    def test_fit(self):
        numpy.random.seed(0)
        nlm = NLM(n_coords = 2, mapping_kind = None)
        assert(nlm.fit(samples[:3]) == nlm)
        assert(hasattr(nlm, 'embedding_'))
        assert(nlm.embedding_.shape == (3, 2))
        assert_array_almost_equal(dist2hd(nlm.embedding_[:3],
            nlm.embedding_[:3])**2, distances[:3, :3], decimal = 4)

    @raises(RuntimeError)
    def test_transform_raises(self):
        numpy.random.seed(0)
        nlm = NLM(n_coords = 2, mapping_kind = None)
        nlm.fit(samples[:3])
        nlm.transform(samples[0])

    def test_transform(self):
        numpy.random.seed(0)
        nlm = NLM(n_coords = 2, n_neighbors = 3)
        nlm.fit(samples[:3])
        mapped = nlm.transform(samples)
        assert_array_almost_equal(mapped[:3], nlm.embedding_, decimal=3)

if __name__ == "__main__":
  import unittest
  unittest.main()

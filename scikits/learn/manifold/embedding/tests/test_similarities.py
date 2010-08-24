#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy

from unittest import TestCase
from ..similarities import LLE

class test_lle(TestCase):
    def test_lle(self):
        samples = numpy.array((0., 0., 0.,
          1., 0., 0.,
          0., 1., 0.,
          1., 1., 0.,
          0., .5, 0.,
          .5, 0., 0.,
          1., 1., 0.5,
          )).reshape((-1,3))
        lle = LLE(n_coords = 2, n_neighbors=5)
        lle.fit(samples)
        
        assert(lle.embedding_.shape == (7,2))

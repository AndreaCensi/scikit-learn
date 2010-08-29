#!/usr/bin/env python
# -*- coding: utf-8 -*-

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('optimization',parent_package,top_path)
    config.add_subpackage('criterion')
    config.add_subpackage('helpers')
    config.add_subpackage('line_search')
    config.add_subpackage('optimizer')
    config.add_subpackage('step')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

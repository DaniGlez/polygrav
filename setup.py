#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name='polygrav',
      version='0.0.1',
      description='numpy implementation of the polyhedral gravity model from Werner - Scheeres (1996)',
      url='https://github.com/DaniGlez/polygrav',
      author=u'Daniel González Arribas',
      author_email='dangonza@ing.uc3m.es',
      license='MIT',
      packages=['polygrav'],
      install_requires=[
          'numpy',
      ],
      extras_require = {
           'cuda': ['pycuda']
        }
)

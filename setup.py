#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 07:51:39 2022

@author: dean
"""

from setuptools import setup, find_packages

install_requires = [
                        "pandas",
                        "numpy",
                        "swmfio",
                        "vtk"
                    ]

setup(
    name='deltaB',
    version='0.5.0',
    author='Dean Thomas',
    author_email='dean.thomas@physics123.net',
    packages=find_packages(),
    description='Examine delta B contributions from BATSRUS results',
    install_requires=install_requires
)

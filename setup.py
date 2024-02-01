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
                        "scipy",
                        "joblib",
                        "multiprocessing",
                        "matplotlib",
                        "numba",
                        "logging",
                        "seaborn",
                        "cartopy",
                        "spacepy",
                        "swmfio @ git+https://github.com/GaryQ-physics/swmfio.git#egg=swmfio",
                        "magnetopost @ git+https://github.com/GaryQ-physics/magnetopost#egg=magnetopost",
                        "vtk"
                    ]

setup(
    name='deltaB',
    version='1.0.0',
    author='Dean Thomas',
    author_email='dean.thomas@physics123.net',
    packages=find_packages(),
    description='For analyzing Space Weather Model Framework (SWMF) results, using Biot-Savart Law to determine delta B contributions from various currents and geospace regions',
    install_requires=install_requires
)

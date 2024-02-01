#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 17:31:23 2023

@author: Dean Thomas
"""

import os.path

###############################################
# Based on magnetopost info structure
###############################################

data_dir = r'/Volumes/PhysicsHDv3'

info = {
        "model": "SWMF",
        "run_name": "CARR_Scenario1",
        "rCurrents": 1.8,
        "rIonosphere": 1.01725,
        "file_type": "out",
        "dir_run": os.path.join(data_dir, "CARR_Scenario1"),
        "dir_plots": os.path.join(data_dir, "CARR_Scenario1.plots"),
        "dir_derived": os.path.join(data_dir, "CARR_Scenario1.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "CARR_Scenario1", "MAG-3D"),
        "dir_ionosphere": os.path.join(data_dir, "CARR_Scenario1", "IONO-2D")
}

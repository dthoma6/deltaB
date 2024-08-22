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

# data_dir = r'/Users/Shared'
data_dir = r'/Volumes/PhysicsHD'

info = {
        "model": "SWMF",
        "run_name": "Bob_Weigel_070323_3",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "method": "method1",
        "dir_run": os.path.join(data_dir, "Bob_Weigel_070323_3"),
        "dir_plots": os.path.join(data_dir, "Bob_Weigel_070323_3.plots"),
        "dir_derived": os.path.join(data_dir, "Bob_Weigel_070323_3.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "GM_CDF"),
        "dir_ionosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "IONO-2D_CDF")
}

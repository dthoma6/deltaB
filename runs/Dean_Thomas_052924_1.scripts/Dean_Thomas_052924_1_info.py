#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:41:53 2024

@author: Dean Thomas
"""

import os.path

data_dir = r'/Volumes/PhysicsHD'

info = {
        "model": "OpenGGCM",
        "run_name": "Dean_Thomas_052924_1",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "method": "method1",
        "dir_run": os.path.join(data_dir, "Dean_Thomas_052924_1"),
        "dir_plots": os.path.join(data_dir, "Dean_Thomas_052924_1.plots"),
        "dir_derived": os.path.join(data_dir, "Dean_Thomas_052924_1.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "Dean_Thomas_052924_1", "GM_CDF"),
        "dir_ionosphere": os.path.join(data_dir, "Dean_Thomas_052924_1", "IONO-2D_CDF")
}

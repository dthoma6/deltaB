#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:08:07 2023

@author: Dean Thomas
"""

import os
import deltaB as db

# Prior to running this script, execute the following from the directory of
# this script:
#   mkdir runs && cd runs
#   wget -r -np -nH -R "index.html*" --cut-dirs=2 http://mag.gmu.edu/git-data/ccmc/SWPC_SWMF_052811_2/
#   cd SWPC_SWMF_052811_2; find . -name "*.cdf.gz" | xargs -J{} gunzip {}

data_dir = '/Users/dean/Documents/GitHub/magnetopost/runs'

# TODO: Get rCurrents from file
info = {
        "model": "SWMF",
        "run_name": "SWPC_SWMF_052811_2",
        "rCurrents": 4.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
        "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
        "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
        "deltaB_files": {
            "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
        }
}

# Locations to compute B. See config.py for list of known points.
# points  = ["YKC", "GMpoint1"]
points  = ["YKC"]

reduce = False

for i in range(len(points)): 
    # db.loop_ms_b(info, points[i], reduce)    
    db.loop_gap_b(info, points[i], reduce, 30, 30, 30)
    # db.loop_iono_b(info, points[i], reduce, 30, 30, 30)

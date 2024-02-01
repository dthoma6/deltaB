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

# Also edit data_dir in SWPC_SWMF_052811_2_info.py to specify where files
# calculated above are stored
from SWPC_SWMF_052811_2_info import info

# Locations to compute B. See config.py for list of known points.
points  = ["YKC"]

reduce = None

# Get a list of BATSRUS and RIM files, info parameters define location 
# (dir_run) and file types.  See definition of info = {...} above.
from magnetopost import util as util
util.setup(info)

for i in range(len(points)): 
    db.loop_ms_b(info, points[i], reduce)    
    db.loop_gap_b(info, points[i], reduce, nR=100, useRIM=True)
    db.loop_iono_b(info, points[i], reduce)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:12:48 2024

@author: Dean Thomas
"""

import magnetopost as mp

# Prior to running this script, execute the following from the directory of
# this script
#
# mkdir runs
#  or
# mkdir /Big/Disk/runs; ln -s /Big/Disk/runs
#
# cd runs
# wget -r -np -nH --cut-dirs=2 http://mag.gmu.edu/git-data/dwelling/divB_simple1
#
# Then modify dir_run in the following dictionary

from Bob_Weigel_070323_3_info import info as info

points  = ["Colaba"] # Locations to compute B. See config.py for list of known points.
compute = True
plot    = True
n_steps = None       # If None, process all files

# Create output dirs if needed and list of files to process
mp.util.setup(info)

if compute:
    #mp.postproc.job_ie(info, points, n_steps=n_steps)
    mp.postproc.job_ms(info, points, n_steps=n_steps)

if plot:
    for point in points:
        mp.plot.surf_point(info, point, n_steps=n_steps)
        mp.plot.msph_point(info, point, n_steps=n_steps)

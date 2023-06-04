#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:38:00 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db

#################################################################
#
# Example script for generating 2D Bn (B north) versus time,
# Be (B east) versus time, and Bd (B down) versus time plots.
# Results are provided for the magnetosphere based on BATS-R-US data
# and for the gap region and the ionosphere based on RIM data  
#
#################################################################

# info tells the script where the data files are stored and where
# to save plots and calculated data

data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

info = {
        "model": "SWMF",
        "run_name": "DIPTSUR2",
        "rCurrents": 4.0,
        "rIonosphere": 1.01725,
        "file_type": "out",
        "dir_run": os.path.join(data_dir, "DIPTSUR2"),
        "dir_plots": os.path.join(data_dir, "DIPTSUR2.plots"),
        "dir_derived": os.path.join(data_dir, "DIPTSUR2.derived"),
}

if __name__ == "__main__":

    from magnetopost import util as util
    util.setup(info)
    
    # Locations to compute B. See config.py for list of known points.
    # points  = ["YKC", "GMpoint1"]
    points  = ["Colaba"]
    
    # Do we skip files to save time.  If None, do all files.  If not
    # None, then reduce is an integer that determine how many files are skipped
    # e.g., do every 10th file
    reduce = None
    
    # Calculate the delta B sums to get Bn, Be,and Bd contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere.  Bn, Be, and Bd calcuated at points[0]
    db.loop_ms_b(info, points[0], reduce)    
    db.loop_gap_b(info, points[0], reduce, 30, 30, 30)
    db.loop_iono_b(info, points[0], reduce, 30, 30, 30)
    
    # Plot the results
    db.plot_Bned_ms_gap_iono(info, points[0])

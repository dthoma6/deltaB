#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import os.path
from deltaB import loop_2D_ms, plot_2D_ms, \
    loop_2D_gap_iono, plot_2D_gap_iono, \
    plot_2D_ms_gap_iono


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
 
    # Point in SM coordinates where delta B contributions will be calculated
    XSM=[1,0,0]
    
    # Limits for time (x-axis) and Bnorth (y-axis) on plots
    TIME_LIMITS = [4,16]
    BN_LIMITS = [-1200,400]

    # Do we skip files to save time.  If None, do all files.  If not
    # None, then reduce is an integer that determine how many files are skipped
    reduce = None

    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)
    
    # Calculate the delta B sums to get Bn contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere
    loop_2D_ms(XSM, info, reduce)
    loop_2D_gap_iono(XSM, info, reduce)

    # Create 2d plots of Bn vs. time
    plot_2D_ms( info, TIME_LIMITS, BN_LIMITS )
    plot_2D_gap_iono( info, TIME_LIMITS, BN_LIMITS )
    plot_2D_ms_gap_iono( info, TIME_LIMITS, BN_LIMITS )

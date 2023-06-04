#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import os.path
from deltaB import loop_heatmap_ms, plot_heatmap_ms, \
    loop_heatmap_iono, plot_heatmap_iono, \
    loop_heatmap_gap, plot_heatmap_gap

#################################################################
#
# Example script for generating Bn (B north) heatmaps that show how
# Bn varies over the surface of the earth.  Results are provided 
# for the magnetosphere based on BATS-R-US data and for the gap region 
# and the ionosphere based on RIM data.
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
    
    # Max/min of scale used in heatmaps
    VMIN = -1500.
    VMAX = 1500.

    # We will plot the magnitude of the B field in a lat/long grid
    # Define the grid size
    NLAT = 9
    NLONG = 12

    # The times for the files that we will process to create heatmaps
    TIMES = ((2019, 9, 2, 4, 15, 0), 
             (2019, 9, 2, 6, 30, 0),
             (2019, 9, 2, 10, 30, 0))

    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)
    
    # Calculate the delta B sums to get Bn contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere over a lat-long grid
    loop_heatmap_ms( info, TIMES, NLAT, NLONG )
    loop_heatmap_iono( info, TIMES, NLAT, NLONG )
    loop_heatmap_gap( info, TIMES, NLAT, NLONG )

    # Create heatmaps plots of Bn over earth
    plot_heatmap_ms( info, TIMES, VMIN, VMAX, NLAT, NLONG )
    plot_heatmap_iono( info, TIMES, VMIN, VMAX, NLAT, NLONG )
    plot_heatmap_gap( info, TIMES, VMIN, VMAX, NLAT, NLONG )

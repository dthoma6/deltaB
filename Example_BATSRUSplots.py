#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import os.path
from deltaB import loop_2D_BATSRUS, \
    loop_2D_BATSRUS_with_cuts, \
    loop_2D_BATSRUS_3d_cut_vtk, \
    loop_2D_BATSRUS_3d_cut_plots


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

limits = { 
    "RLOG_LIMITS": [1, 1000],
    "R_LIMITS": [0, 300],
    'RHO_LIMITS': [10**-2, 10**4],
    'P_LIMITS': [10**-5, 10**3],
    'JMAG_LIMITS': [10**-11, 10**1],
    'J_LIMITS': [-1, 1],
    'JCDF_LIMITS': [-0.1, 0.1],
    'UMAG_LIMITS': [10**-3, 10**4],
    'U_LIMITS': [-1100, 1100],
    'DBNORM_LIMITS': [10**-15, 10**-1],
    'DBX_SUM_LIMITS': [-1500, 1500],
    'DBY_SUM_LIMITS': [-1500, 1500],
    'DBZ_SUM_LIMITS': [-1500, 1500],
    'DBP_SUM_LIMITS': [-1500, 1500],
    'DB_SUM_LIMITS': [0, 1500],
    'DB_SUM_LIMITS2': [-1200,400],
    'PLOT3D_LIMITS': [-10, 10],
    'XYZ_LIMITS': [-300, 300],
    'XYZ_LIMITS_SMALL': [-20, 20],
    'TIME_LIMITS': [4,16],
    'FRAC_LIMITS': [-0.5,1.5],
    'VMIN': 0.02,
    'VMAX': 0.5
}

cuts = {
    'CUT1_JRMIN': 0.02,
    'CUT2_JPHIMIN': 0.02,
    'CUT2_RMIN': 5,
    'CUT3_JPHIMIN': 0.02
}


if __name__ == "__main__":

    # Point in GSM coordinates where delta B contributions will be calculated
    XGSM=[1,0,0]
   
    # Do we skip files to save time.  If None, do all files.  If not
    # None, then its a integer that determines how many files are skipped
    reduce = 200
    
    # If we make a cut on the data, which cut to make
    cut_selected = 3

    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)
    
    # Generate  various plots of BATSRUS data
    loop_2D_BATSRUS(XGSM, info, limits, reduce)
    # loop_2D_BATSRUS_with_cuts(XGSM, info, limits, cuts, cut_selected, reduce)
    # loop_2D_BATSRUS_3d_cut_vtk(XGSM, info, limits, cuts, reduce)
    # loop_2D_BATSRUS_3d_cut_plots(XGSM, info, limits, cuts, reduce)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:09:48 2023

@author: Dean Thomas
"""

####################################################################
####################################################################
#
# Comparison of CCMC, Gary's and my methods of calculating total
# B at a specific point on earth.  B is sum of Biot-Savart contributions
# from each cell in BATSRUS grid  for r > rCurrents
#
# Initial analysis suggested that different methods of performing
# coordinate transformations (e.g., GEO to GSM) may be causes
# some of the differences in results.  This uses two results from two
# runs of my code.  One using spacepy and one using geopack for 
# coordinate transforms.  Observed differences were order of a few nT.
#
####################################################################
####################################################################

import magnetopost as mp
import os.path
import pandas as pd
import deltaB as db
import matplotlib.pyplot as plt
import numpy as np

def surf_point(info, surface_locations, n_steps=None):

    # Code borrowed from Gary to read CMCC and magnetopost files
    if isinstance(surface_locations, list):
        for surface_location in surface_locations:
            print(surface_location)
            surf_point(info, surface_location, n_steps=n_steps)
        return
    else:
        surface_location = surface_locations
    # End Gary's code

    # Read my data, first the spacepy results, then geopack method
    pklname = 'dB_bs_msph-' + surface_location + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    pklname = 'dB_bs_msph-' + surface_location + '_geopack.pkl'
    df2 = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the two methods
    df['mag']      = np.sqrt(df['Bn']**2         + df['Be']**2        + df['Bd']**2)
    df2['mag']     = np.sqrt(df2['Bn']**2        + df2['Be']**2       + df2['Bd']**2)
    
    # Combine the two dataframes into one, first renaming the columns to avoid
    # collisons.
    df.columns =['Year_sp', 'Month_sp', 'Day_sp', 'Hour_sp', 'Minute_sp', 'Second_sp', 'Bn sp', 'Be sp', 'Bd sp', 'Bx sp', 'By sp', 'Bz sp', 'mag sp']
    df2.columns =['Year_ge', 'Month_ge', 'Day_ge', 'Hour_ge', 'Minute_ge', 'Second_ge', 'Bn ge', 'Be ge', 'Bd ge','Bx ge', 'By ge', 'Bz ge', 'mag ge']
    df_all = df.merge( df2, left_index=True, right_index=True)
    
    # Calculate differences between the B fields from the three methods.
    df_all['sp-ge north'] = df_all['Bn sp'] - df_all['Bn ge']
    df_all['sp-ge east'] = df_all['Be sp'] - df_all['Be ge']
    df_all['sp-ge down'] = df_all['Bd sp'] - df_all['Bd ge']
    df_all['sp-ge mag'] = df_all['mag sp'] - df_all['mag ge']
 
    ax = df_all.plot( y='sp-ge north', label='spacepy-geopack north')
    ax.set_ylabel( r'$\Delta B_N$' )

    ax = df_all.plot( y='sp-ge east', label='spacepy-geopack east')
    ax.set_ylabel( r'$\Delta B_E$' )

    ax = df_all.plot( y='sp-ge down', label='spacepy-geopack down')
    ax.set_ylabel( r'$\Delta B_D$' )

    ax = df_all.plot( y='sp-ge mag', label='spacepy-geopack mag')
    ax.set_ylabel( r'$| B |$' )

    print('Mean sp-ge north: ', df_all['sp-ge north'].mean())
    print('Mean sp-ge east: ', df_all['sp-ge east'].mean())
    print('Mean sp-ge down: ', df_all['sp-ge down'].mean())
    print('Mean sp-ge mag: ', df_all['sp-ge mag'].mean())

data_dir = '/Users/dean/Documents/GitHub/magnetopost/runs'

# TODO: Get rCurrents from file
info = {
        "model": "SWMF",
        "run_name": "SWPC_SWMF_052811_2",
        "rCurrents": 4.0,
        "file_type": "cdf",
        "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
        "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
        "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
        "deltaB_files": {
            "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
        }
}

from magnetopost import util as util
util.setup(info)

# Locations to compute B. See config.py for list of known points.
# points  = ["YKC", "GMpoint1"]
points  = ["YKC"]
# points  = ["GMpoint1"]

surf_point(info, points, n_steps=None)

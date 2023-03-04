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
# from each cell in BATSRUS grid for r > rCurrents
#
# See SWPC_SWMF_052811_2_horz.py and SWPC_SWMF_052811_2_diff_coord.py
# for additional results.  Bottom line conclusion is that different
# methods of doing coordinate transformations and differences in the
# decomposition of Total B into north-east-down results.
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

    if 'deltaB_files' in info and surface_location in info['deltaB_files']:
        if info['file_type'] == "cdf":
            try:
                dBMhd, dBFac, dBHal, dBPed = mp.extract_magnetometer_data.extract_from_swmf_ccmc_printout_file(info, surface_location, n_steps=n_steps)
                mp.logger.info("Found SWMF/CCMC magnetometer file for " + surface_location)
                compare = True
                label = 'CCMC'
            except:
                mp.logger.info("Did not find SWMF/CCMF magnetometer file for " + surface_location)
        if info['file_type'] == "out":
            try:
                dBMhd, dBFac, dBHal, dBPed = mp.extract_magnetometer_data.extract_from_swmf_magnetometer_files(info, surface_location, n_steps=n_steps)
                mp.logger.info("Found SWMF magnetometer grid data file")
                label = 'SWMF'
                compare = True
            except:
                mp.logger.info("SWMF magnetometer grid data file not found")
    

    bs_msph, bs_fac, bs_hall, bs_pedersen, cl_gap, helm_outer, helm_rCurrents_gapSM, probe = mp.extract_magnetometer_data.extract_from_magnetopost_files(info, surface_location, n_steps=n_steps)
    # End of Gary's code

    # Read my data
    pklname = 'dB_bs_iono-' + surface_location + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['Pmag']     = np.sqrt(df['Bnp']**2         + df['Bep']**2        + df['Bdp']**2)
    bs_pedersen['Pmag'] = np.sqrt(bs_pedersen['north']**2  + bs_pedersen['east']**2  + bs_pedersen['down']**2)
    dBPed['Pmag']  = np.sqrt(dBPed['north']**2   + dBPed['east']**2   + dBPed['down']**2)
    
    df['Hmag']     = np.sqrt(df['Bnh']**2         + df['Beh']**2        + df['Bdh']**2)
    bs_hall['Hmag'] = np.sqrt(bs_hall['north']**2  + bs_hall['east']**2  + bs_hall['down']**2)
    dBHal['Hmag']  = np.sqrt(dBHal['north']**2   + dBHal['east']**2   + dBHal['down']**2)
    
    # Combine the three dataframes into one.  
    #
    # NOTE, all three dataframes use the file timestamps to index the entries, 
    # but the CCMC file (dBMhd) has more points than the other two.  
    #
    # The other two dataframes (Gary's and mine) have the same number of
    # entries.  The code below joins them on matching indices.
    #
    # CCMC starts with 2460 points, Gary with 2191, and Mine with 2191.  The 
    # combined dataframe, df_all, has 1546 points.
    #
    bs_pedersen.columns =['p north_bs','p east_bs','p down_bs','p mag_bs']
    dBPed.columns =['p north_db','p east_db','p down_db','p mag_db']
    df_allp = bs_pedersen.merge( dBPed, left_index=True, right_index=True)
    df_allp = df_allp.merge( df, left_index=True, right_index=True)
    
    bs_hall.columns =['h north_bs','h east_bs','h down_bs','h mag_bs']
    dBHal.columns =['h north_db','h east_db','h down_db','h mag_db']
    df_allh = bs_hall.merge( dBHal, left_index=True, right_index=True)
    df_allh = df_allh.merge( df, left_index=True, right_index=True)
       
    # Plot results
    ax = df_allp.plot( y='p north_bs', label='North Gary')
    df_allp.plot(ax = ax, y='p north_db', label='North CCMC')
    df_allp.plot( ax=ax, y='Bnp', label='North Dean', ls='dotted')
    ax.set_ylabel( r'$B_N$ Pedersen' )
     
    ax = df_allp.plot( y='p east_bs', label='East Gary')
    df_allp.plot(ax = ax, y='p east_db', label='East CCMC')
    df_allp.plot( ax=ax, y='Bep', label='East Dean', ls='dotted')
    ax.set_ylabel( r'$B_E$ Pedersen' )
    
    ax = df_allp.plot( y='p down_bs', label='Down Gary')
    df_allp.plot(ax = ax, y='p down_db', label='Down CCMC')
    df_allp.plot( ax=ax, y='Bdp', label='Down Dean', ls='dotted')
    ax.set_ylabel( r'$B_D$ Pedersen' )
 
    ax = df_allh.plot( y='h north_bs', label='North Gary')
    df_allh.plot(ax = ax, y='h north_db', label='North CCMC')
    df_allh.plot( ax=ax, y='Bnh', label='North Dean', ls='dotted')
    ax.set_ylabel( r'$B_N$ Hall' )
     
    ax = df_allh.plot( y='h east_bs', label='East Gary')
    df_allh.plot(ax = ax, y='h east_db', label='East CCMC')
    df_allh.plot( ax=ax, y='Beh', label='East Dean', ls='dotted')
    ax.set_ylabel( r'$B_E$ Hall' )
    
    ax = df_allh.plot( y='h down_bs', label='Down Gary')
    df_allh.plot(ax = ax, y='h down_db', label='Down CCMC')
    df_allh.plot( ax=ax, y='Bdh', label='Down Dean', ls='dotted')
    ax.set_ylabel( r'$B_D$ Hall' )
    
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

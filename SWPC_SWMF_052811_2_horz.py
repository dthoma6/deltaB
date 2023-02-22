#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 14:09:48 2023

@author: Dean Thomas
"""
import magnetopost as mp
import os.path
import pandas as pd
import deltaB as db
import matplotlib.pyplot as plt
import numpy as np

####################################################################
####################################################################
#
# Comparison of CCMC, Gary's and my methods of calculating total
# B at a specific point on earth.  B is sum of Biot-Savart contributions
# from each cell in BATSRUS grid  for r > rCurrents
#
# Initial analysis showed differences between my method and CCMC.
# Further analysis suggested that some of the difference was due
# to decomposition of B into north-east-down vectors.  I and CCMC 
# agreed on down, but not east and north.  So I compared down and 
# horzontal.  I and CCMC agree on down and horizontal too within a
# few nT
#
####################################################################
####################################################################


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


    bs_msph, bs_fac, bs_hall, bs_pedersen, cl_msph, helm_outer, helm_rCurrents_gapSM, probe = mp.extract_magnetometer_data.extract_from_magnetopost_files(info, surface_location, n_steps=n_steps)
    # End of Gary's code

    # Read my data
    pklname = 'dB_bs_msph-' + surface_location + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['mag']      = np.sqrt(df['Bn']**2         + df['Be']**2        + df['Bd']**2)
    bs_msph['mag'] = np.sqrt(bs_msph['north']**2 + bs_msph['east']**2 + bs_msph['down']**2)
    dBMhd['mag']   = np.sqrt(dBMhd['north']**2   + dBMhd['east']**2   + dBMhd['down']**2)
    
    # Calculate the magnitude of the horizontal B fields from the three methods
    df['Bh']      = np.sqrt(df['Bn']**2         + df['Be']**2        )
    bs_msph['Bh'] = np.sqrt(bs_msph['north']**2 + bs_msph['east']**2 )
    dBMhd['Bh']   = np.sqrt(dBMhd['north']**2   + dBMhd['east']**2   )
    
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
    bs_msph.columns =['north_bs','east_bs','down_bs','mag_bs', 'Bh_bs']
    dBMhd.columns =['north_db','east_db','down_db','mag_db', 'Bh_db']
    df_all = bs_msph.merge( dBMhd, left_index=True, right_index=True)
    df_all = df_all.merge( df, left_index=True, right_index=True)
    
    # Calculate differences between the B fields from the three methods.
    df_all['db-bs north'] = df_all['north_db'] - df_all['north_bs']
    df_all['db-bs east'] = df_all['east_db'] - df_all['east_bs']
    df_all['db-bs down'] = df_all['down_db'] - df_all['down_bs']
 
    df_all['db-mine north'] = df_all['north_db'] - df_all['Bn']
    df_all['db-mine east'] = df_all['east_db'] - df_all['Be']
    df_all['db-mine down'] = df_all['down_db'] - df_all['Bd']
 
    df_all['bs-mine north'] = df_all['north_bs'] - df_all['Bn']
    df_all['bs-mine east'] = df_all['east_bs'] - df_all['Be']
    df_all['bs-mine down'] = df_all['down_bs'] - df_all['Bd']
 
    df_all['db-bs mag'] = df_all['mag_db'] - df_all['mag_bs']
    df_all['db-mine mag'] = df_all['mag_db'] - df_all['mag']
    df_all['bs-mine mag'] = df_all['mag_bs'] - df_all['mag']
 
    df_all['db-bs Bh'] = df_all['Bh_db'] - df_all['Bh_bs']
    df_all['db-mine Bh'] = df_all['Bh_db'] - df_all['Bh']
    df_all['bs-mine Bh'] = df_all['Bh_bs'] - df_all['Bh']
 
    # Plot results
    ax = df_all.plot( y='north_bs', label='North Gary')
    df_all.plot(ax = ax, y='north_db', label='North CCMC')
    df_all.plot( ax=ax, y='Bn', label='North Dean')
    ax.set_ylabel( r'$B_N$' )
    
    ax = df_all.plot( y='east_bs', label='East Gary')
    df_all.plot(ax = ax, y='east_db', label='East CCMC')
    df_all.plot( ax=ax, y='Be', label='East Dean')
    ax.set_ylabel( r'$B_E$' )
    
    ax = df_all.plot( y='down_bs', label='Down Gary')
    df_all.plot(ax = ax, y='down_db', label='Down CCMC')
    df_all.plot( ax=ax, y='Bd', label='Down Dean')
    ax.set_ylabel( r'$B_D$' )

    ax = df_all.plot( y='Bh_bs', label='Horz Gary')
    df_all.plot(ax = ax, y='Bh_db', label='Horz CCMC')
    df_all.plot( ax=ax, y='Bh', label='Horz Dean')
    ax.set_ylabel( r'|$B_H|$' )
 
    ax = df_all.plot( y='db-bs north', label='CCMC-Gary north')
    df_all.plot(ax = ax, y='db-mine north', label='CCMC-Dean north')
    df_all.plot( ax=ax, y='bs-mine north', label='Gary-Dean north')
    ax.set_ylabel( r'$\Delta B_N$' )

    ax = df_all.plot( y='db-bs east', label='CCMC-Gary east')
    df_all.plot(ax = ax, y='db-mine east', label='CCMC-Dean east')
    df_all.plot( ax=ax, y='bs-mine east', label='Gary-Dean east')
    ax.set_ylabel( r'$\Delta B_E$' )

    ax = df_all.plot( y='db-bs down', label='CCMC-Gary down')
    df_all.plot(ax = ax, y='db-mine down', label='CCMC-Dean down')
    df_all.plot( ax=ax, y='bs-mine down', label='Gary-Dean down')
    ax.set_ylabel( r'$\Delta B_D$' )

    ax = df_all.plot( y='db-bs mag', label='CCMC-Gary')
    df_all.plot(ax = ax, y='db-mine mag', label='CCMC-Dean')
    df_all.plot( ax=ax, y='bs-mine mag', label='Gary-Dean')
    ax.set_ylabel( r'$|B|$' )

    ax = df_all.plot( y='db-bs Bh', label='CCMC-Gary')
    df_all.plot(ax = ax, y='db-mine Bh', label='CCMC-Dean')
    df_all.plot( ax=ax, y='bs-mine Bh', label='Gary-Dean')
    ax.set_ylabel( r'$\Delta |B_H|$' )


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

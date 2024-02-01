#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:38:00 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#################################################################
#
# Script for generating 2D Bn (B north) versus time,
# Be (B east) versus time, and Bd (B down) versus time plots.
# Results are provided for the magnetosphere based on BATS-R-US data
# and for the gap region and the ionosphere based on RIM data  
#
#################################################################

from CARR_Scenarios1_info import info as info

if __name__ == "__main__":

    from magnetopost import util as util
    util.setup(info)
    
    # Location to compute B. See config.py for list of known points.
    point = "Colaba"
    
    # Do we skip files to save time.  If None, do all files.  If not
    # None, then reduce is an integer that determine how many files are skipped
    # e.g., do every 10th file
    reduce = None
    
    # Calculate the delta B sums to get Bn, Be,and Bd contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere.  Bn, Be, and Bd calcuated at points[0]
    db.loop_ms_b(info, point, reduce)    
    db.loop_gap_b(info, point, reduce, nR=100, useRIM=True)
    db.loop_iono_b(info, point, reduce)
    
    # # Plot the results
    # db.plot_Bned_ms_gap_iono(info, point)

    # Create custom plot for paper
    
    # Set some plot configs
    plt.rcParams["figure.figsize"] = [6,4.5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    # Read magnetosphere, gap region, and ionosphere data
    # Rename columns to make tidy names on plots
    # Divide ionosphere data into Pedersen and Hall currents
    pklname = 'dB_bs_msph-' + point + '.pkl'
    df_ms = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ms.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
 
    pklname = 'dB_bs_gap-' + point + '.pkl'
    df_gap = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_gap.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    pklname = 'dB_bs_iono-' + point + '.pkl'
    df_iono = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ped = deepcopy(df_iono)
    df_ped.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'$B_Nh$', r'$B_Eh$', r'$B_Dh$', r'$B_xh$', r'$B_yh$', r'$B_zh$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    df_hall = deepcopy(df_iono)
    df_hall.columns = [r'$B_Np$', r'$B_Ep$', r'$B_Dp$', r'$B_xp$', r'$B_yp$', r'$B_zp$', \
                      r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    # Calculate total Bn, total Be, total Bh
    df_ms[r'$B_{N}$ $Total$'] = df_ms[r'$B_N$'] + df_gap[r'$B_N$'] + df_ped[r'$B_N$'] + df_hall[r'$B_N$']
    df_ms[r'$B_{E}$ $Total$'] = df_ms[r'$B_E$'] + df_gap[r'$B_E$'] + df_ped[r'$B_E$'] + df_hall[r'$B_E$']
    df_ms[r'$B_{H}$ $Total$'] = np.sqrt( df_ms[r'$B_{N}$ $Total$']**2 + df_ms[r'$B_{E}$ $Total$']**2 )
    df_ms[r'$|B_{N}| - |B_{H}|$ $Total$'] = np.abs(df_ms[r'$B_{N}$ $Total$']) - df_ms[r'$B_{H}$ $Total$']
    df_ms[r'ratio'] = np.abs(df_ms[r'$B_{H}$ $Total$'] / df_ms[r'$B_{N}$ $Total$'])
        
    # Create directory for plots
    db.create_directory( info['dir_plots'], 'Bned'  )
    
    # Create plots and save them 
    ax = df_ms.plot(x=r'Time (hr)', y=[r'$B_{N}$ $Total$', r'$|B_{N}| - |B_{H}|$ $Total$'],\
                legend=True, \
                style=['-','r:','--'], \
                grid = False,\
                xlim = [4.5,9.5],\
                ylabel = r'B-field (nT) at ' + point, xlabel = 'Time (UTC)', \
                title = 'Scenario 1 at ' + point)
    # plt.xticks(ticks=[4,6,8,10,12,14,16],labels=['04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00']) 
    plt.xticks(ticks=[5,6,7,8,9],labels=['05:00', '06:00', '07:00', '08:00', '09:00']) 
        
    pltname = 'tot-Bn-Bh-diff-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.svg' ) )

    ax = df_ms.plot(x=r'Time (hr)', y=[r'ratio'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                xlim = [4.5,9.5],\
                ylabel = r'$|B_{H} / B_{N}|$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Scenario 1 at ' + point)
    # plt.xticks(ticks=[4,6,8,10,12,14,16],labels=['04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00']) 
    plt.xticks(ticks=[5,6,7,8,9],labels=['05:00', '06:00', '07:00', '08:00', '09:00']) 

    pltname = 'tot-Bn-Bh-ratio-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'Bned', pltname + '.svg' ) )

    # Get some statistics on the ratio of Bh and Bn
    # Be sure to look at stats only between 4.5 hours and 9.5 hours
    df_ms = df_ms.drop(df_ms[df_ms[r'Time (hr)'] > 9.5].index)
    df_ms = df_ms.drop(df_ms[df_ms[r'Time (hr)'] < 4.5].index)
    mean = np.mean(df_ms[r'ratio'])
    stdev = np.std(df_ms[r'ratio'])
    
    print('Mean and stdev of ratio: ', mean, stdev)
    

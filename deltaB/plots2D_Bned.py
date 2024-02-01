#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:39 2023

@author: Dean Thomas
"""

import os.path
import pandas as pd
from copy import deepcopy
import matplotlib as plt
from deltaB import create_directory

# data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

# info = {
#         "model": "SWMF",
#         "run_name": "divB_simple1",
#         "rCurrents": 3.0,
#         "rIonosphere": 1.01725,
#         "file_type": "out",
#         "dir_run": os.path.join(data_dir, "divB_simple1"),
#         "dir_plots": os.path.join(data_dir, "divB_simple1.plots"),
#         "dir_derived": os.path.join(data_dir, "divB_simple1.derived"),
#         "dir_magnetosphere": os.path.join(data_dir, "divB_simple1", "GM"),
#         "dir_ionosphere": os.path.join(data_dir, "divB_simple1", "IE")

def plot_Bned_ms_gap_iono(info, obs_point):
    """ 2D plots of north-east-down components of B field due to magnetosphere,
    gap region, Pedersen, and Hall currents.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above

       ob s_point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
    
    Outputs:
        None = other than plots generated
    """
    
    # Read magnetosphere, gap region, and ionosphere data
    # Rename columns to make tidy names on plots
    # Divide ionosphere data into Pedersen and Hall currents
    pklname = 'dB_bs_msph-' + obs_point + '.pkl'
    df_ms = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ms.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
 
    pklname = 'dB_bs_gap-' + obs_point + '.pkl'
    df_gap = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_gap.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    pklname = 'dB_bs_iono-' + obs_point + '.pkl'
    df_iono = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ped = deepcopy(df_iono)
    df_ped.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'$B_Nh$', r'$B_Eh$', r'$B_Dh$', r'$B_xh$', r'$B_yh$', r'$B_zh$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    df_hall = deepcopy(df_iono)
    df_hall.columns = [r'$B_Np$', r'$B_Ep$', r'$B_Dp$', r'$B_xp$', r'$B_yp$', r'$B_zp$', \
                      r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    # Create directory for plots
    create_directory( info['dir_plots'], 'Bned'  )
    
    # Create plots and save them
    fig = df_ms.plot(y=[r'$B_N$', r'$B_E$', r'$B_D$'], use_index=True, \
               ylabel = 'B field component (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Magnetosphere ' + obs_point ).get_figure()
    pltname = 'ms-Bned-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )
    
    fig = df_gap.plot(y=[r'$B_N$', r'$B_E$', r'$B_D$'], use_index=True, \
               ylabel = 'B field component (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Gap FAC ' + obs_point ).get_figure()
    pltname = 'gap-Bned-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    fig = df_ped.plot(y=[r'$B_N$', r'$B_E$', r'$B_D$'], use_index=True, \
               ylabel = 'B field component (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Pedersen ' + obs_point ).get_figure()
    pltname = 'pedersen-Bned-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    fig = df_hall.plot(y=[r'$B_N$', r'$B_E$', r'$B_D$'], use_index=True, \
               ylabel = 'B field component (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Hall ' + obs_point ).get_figure()
    pltname = 'hall-Bned-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    return

def plot_Bn_ms_gap_iono(info, obs_point):
    """ 2D plots of north component of B field due to magnetosphere,
    gap region, Pedersen, and Hall currents.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above

       ob s_point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
    
    Outputs:
        None = other than plots generated
    """
    
    # Read magnetosphere, gap region, and ionosphere data
    # Rename columns to make tidy names on plots
    # Divide ionosphere data into Pedersen and Hall currents
    pklname = 'dB_bs_msph-' + obs_point + '.pkl'
    df_ms = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ms.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
 
    pklname = 'dB_bs_gap-' + obs_point + '.pkl'
    df_gap = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_gap.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                     r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    pklname = 'dB_bs_iono-' + obs_point + '.pkl'
    df_iono = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_ped = deepcopy(df_iono)
    df_ped.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'$B_Nh$', r'$B_Eh$', r'$B_Dh$', r'$B_xh$', r'$B_yh$', r'$B_zh$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    df_hall = deepcopy(df_iono)
    df_hall.columns = [r'$B_Np$', r'$B_Ep$', r'$B_Dp$', r'$B_xp$', r'$B_yp$', r'$B_zp$', \
                      r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    # Create directory for plots
    create_directory( info['dir_plots'], 'Bned'  )
    
    # Create plots and save them
    fig = df_ms.plot(y=r'$B_N$', use_index=True, \
               ylabel = '$B_N$ (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Magnetosphere ' + obs_point ).get_figure()
    pltname = 'ms-Bn-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )
    
    fig = df_gap.plot(y=r'$B_N$', use_index=True, \
               ylabel = '$B_N$ (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Gap FAC ' + obs_point ).get_figure()
    pltname = 'gap-Bn-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    fig = df_ped.plot(y=r'$B_N$', use_index=True, \
               ylabel = '$B_N$ (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Pedersen ' + obs_point ).get_figure()
    pltname = 'pedersen-Bn-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    fig = df_hall.plot(y=r'$B_N$', use_index=True, \
               ylabel = '$B_N$ (nT)', xlabel = 'Time', \
               title = info['run_name'] + ' Hall ' + obs_point ).get_figure()
    pltname = 'hall-Bn-' + obs_point + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'Bned', pltname ) )

    return
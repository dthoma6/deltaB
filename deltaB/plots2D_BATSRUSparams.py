#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 13:26:44 2023

@author: Dean Thomas
"""

import os.path
import logging
import swmfio
import numpy as np
import pandas as pd
from datetime import datetime

from deltaB import create_directory

logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

# info = {...} example is below

# data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

# info = {
#         "model": "SWMF",
#         "run_name": "DIPTSUR2",
#         "rCurrents": 4.0,
#         "rIonosphere": 1.01725,
#         "file_type": "out",
#         "dir_run": os.path.join(data_dir, "DIPTSUR2"),
#         "dir_plots": os.path.join(data_dir, "DIPTSUR2.plots"),
#         "dir_derived": os.path.join(data_dir, "DIPTSUR2.derived"),
# }

def loop_2D_BATSRUSparams(XGSM, info, reduce):
    """Loop thru data in BATSRUS files to create a variety of 2D plots 
    showing various parameters at point XGSM.  The goal is to see things
    like shock waves passing through the point
    
    Inputs:
        XGSM = cartesian position where params will be assessed (GSM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        limits = axis limits on plots, see limits = {...} example above
        
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        None - other than the plot files saved
    """

    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    dtimes = [datetime(*time) for time in times]

    bx = np.zeros(n)
    by = np.zeros(n)
    bz = np.zeros(n)

    jx = np.zeros(n)
    jy = np.zeros(n)
    jz = np.zeros(n)

    ux = np.zeros(n)
    uy = np.zeros(n)
    uz = np.zeros(n)

    e = np.zeros(n)
    p = np.zeros(n)
    rho = np.zeros(n)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
 
        logging.info(f'Data for BASTRUS file... {base}')

        # Read BATSRUS file
        batsrus = swmfio.read_batsrus(filepath)
        assert(batsrus != None)
         
        # Extract data from BATSRUS
        var_dict = dict(batsrus.varidx)
        
        bx[i] = batsrus.interpolate( XGSM, 'bx' )
        by[i] = batsrus.interpolate( XGSM, 'by' )
        bz[i] = batsrus.interpolate( XGSM, 'bz' )

        jx[i] = batsrus.interpolate( XGSM, 'jx' )
        jy[i] = batsrus.interpolate( XGSM, 'jy' )
        jz[i] = batsrus.interpolate( XGSM, 'jz' )

        ux[i] = batsrus.interpolate( XGSM, 'ux' )
        uy[i] = batsrus.interpolate( XGSM, 'uy' )
        uz[i] = batsrus.interpolate( XGSM, 'uz' )

        e[i] = batsrus.interpolate( XGSM, 'e' )
        p[i] = batsrus.interpolate( XGSM, 'p' )
        rho[i] = batsrus.interpolate( XGSM, 'rho' )

    
    df = pd.DataFrame()
    
    df[r'Time'] = dtimes

    df[r'$B_x$'] = bx
    df[r'$B_y$'] = by
    df[r'$B_z$'] = bz
    
    df[r'$|B|$'] = np.sqrt( df[r'$B_x$']**2 + df[r'$B_y$']**2 + df[r'$B_z$']**2 )

    df[r'$j_x$'] = jx
    df[r'$j_y$'] = jy
    df[r'$j_z$'] = jz
    
    df[r'$|j|$'] = np.sqrt( df[r'$j_x$']**2 + df[r'$j_y$']**2 + df[r'$j_z$']**2 )
    
    df[r'$j_{\parallel}$'] = ( df[r'$j_x$'] * df[r'$B_x$'] + \
                              df[r'$j_y$'] * df[r'$B_y$'] + \
                              df[r'$j_z$'] * df[r'$B_z$'] ) / df[r'$|B|$']

    df[r'$j_{\perp}$'] = np.sqrt( df[r'$|j|$']**2 -  df[r'$j_{\parallel}$']**2 )
    
    df[r'$u_x$'] = ux
    df[r'$u_y$'] = uy
    df[r'$u_z$'] = uz

    df[r'$e$'] = e
    df[r'$p$'] = p
    df[r'$\rho$'] = rho
    
    df.set_index(r'Time', inplace=True)

    create_directory(info['dir_plots'], 'params/')
    
    fig = df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = r'Magnetic Field $(nT)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-b-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$j_x$', r'$j_y$', r'$j_z$'], use_index=True, \
                ylabel = r'Current Density $(\mu A / m^2)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-jxyz-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$j_{\parallel}$', r'$j_{\perp}$'], use_index=True, \
                ylabel = r'Current Density $(\mu A / m^2)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-jpp-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-u-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$e$'], use_index=True, \
                ylabel = 'Energy Density $(J / m^3)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-e-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-p-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    fig = df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = r'$\rho$ $Mp/cc$ or $amu / cm^3$ ??', xlabel = 'Time',
                title = info['run_name'] + ' at GSM ' + str(XGSM) ).get_figure()
    pltname = 'ms-rho-' + str(XGSM) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'params', pltname ) )

    return

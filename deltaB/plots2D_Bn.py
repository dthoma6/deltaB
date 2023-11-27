#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: Dean Thomas
"""

import logging
import pandas as pd
import os.path
from datetime import datetime, timedelta

from deltaB import calc_ms_b_paraperp, \
    calc_gap_b, calc_iono_b, \
    convert_BATSRUS_to_dataframe, \
    date_timeISO, create_directory, \
    plotargs_multiy, plot_NxM_multiy, \
    SMtoGSM

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

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

def loop_2D_ms(XSM, info, reduce, delta_hr=None):
    """Loop thru data in BATSRUS files to generate data for 2D plots showing Bn 
    versus time including the breakdown of contributions from currents parallel 
    and perpendicular to local B field.  This routine examines currents in the
    magnetosphere.

    Inputs:
        XSM = cartesian position where magnetic field will be assessed (SM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        delta_hr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the pickle file saved
    """

    # Make sure delta_hr is float
    if delta_hr is not None:
        assert( type(delta_hr) == float )
 
    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Setup temporary variables where results from loop will be saved
    B_ms = [None] * n
    B_ms_parallel = [None] * n
    B_ms_perp = [None] * n
    B_ms_perpphi = [None] * n
    B_ms_perpphires = [None] * n
    B_times = [None] * n
    B_index = [None] * n

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for the BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere dB for 2D... {base}')
    
        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    
        # Record time and index for plots
        if delta_hr is None:
            h = time[3]
            m = time[4]
            B_times[i] = h + m/60
            B_index[i] = i
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=delta_hr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            B_times[i] = h + m/60
            B_index[i] = i
        
        # Convert XSM to GSM coordinates, which is what calc_ms_b_paraperp needs
        XGSM = SMtoGSM(XSM, time, ctype_in='car', ctype_out='car')

        # We want the Bn contributions from the main components of the field - 
        # the complete field, and that due to currents parallel and perpendicular
        # to the local B field (perpphi and perpphires are components of perpendicular)
        B_ms[i], B_ms_parallel[i], B_ms_perp[i], \
            B_ms_perpphi[i], B_ms_perpphires[i] = \
            calc_ms_b_paraperp(XGSM, timeISO, df)

    # Create dataframe from results and save to disk
    dtimes = [datetime(*time) for time in times]

    df = pd.DataFrame( { r'Total': B_ms, 
                        r'Parallel': B_ms_parallel, 
                        r'Perpendicular': B_ms_perp, 
                        r'Perpendicular $\phi$': B_ms_perpphi, 
                        r'Perpendicular Residual': B_ms_perpphires,
                        r'Time (hr)': B_times },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.ms-2D.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def loop_2D_ms_point(point, info, reduce, delta_hr=None):
    """Loop thru data in BATSRUS files to generate data for 2D plots showing Bn 
    versus time including the breakdown of contributions from currents parallel 
    and perpendicular to local B field.  This routine examines currents in the
    magnetosphere.

    Inputs:
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
                    
        info = info on files to be processed, see info = {...} example above
             
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        delta_hr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the pickle file saved
    """
    # Make sure delta_hr is float
    if delta_hr is not None:
        assert( type(delta_hr) == float )
    
    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Setup temporary variables where results from loop will be saved
    B_ms = [None] * n
    B_ms_parallel = [None] * n
    B_ms_perp = [None] * n
    B_ms_perpphi = [None] * n
    B_ms_perpphires = [None] * n
    B_times = [None] * n
    B_index = [None] * n

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for the BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere dB for 2D... {base} at {point}')
    
        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    
        # Record time and index for plots
        if delta_hr is None:
            h = time[3]
            m = time[4]
            B_times[i] = h + m/60
            B_index[i] = i
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time)
            dtime2 = dtime + timedelta(hours=delta_hr)
            timeISO = dtime2.isoformat()
            h = dtime2.hour
            m = dtime2.minute
            B_times[i] = h + m/60
            B_index[i] = i
                
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # BATSRUS data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM2 = XGEO.convert( 'GSM', 'car' )
        XGSM = XGSM2.data[0]

        # We want the Bn contributions from the main components of the field - 
        # the complete field, and that due to currents parallel and perpendicular
        # to the local B field (perpphi and perpphires are components of perpendicular)
        B_ms[i], B_ms_parallel[i], B_ms_perp[i], \
            B_ms_perpphi[i], B_ms_perpphires[i] = \
            calc_ms_b_paraperp(XGSM, timeISO, df)

    # Create dataframe from results and save to disk
    dtimes = [datetime(*time) for time in times]

    df = pd.DataFrame( { r'Total': B_ms, 
                        r'Parallel': B_ms_parallel, 
                        r'Perpendicular': B_ms_perp, 
                        r'Perpendicular $\phi$': B_ms_perpphi, 
                        r'Perpendicular Residual': B_ms_perpphires,
                        r'Time (hr)': B_times },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.ms-2D-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def plot_2D_ms( info, time_limits, Bn_limits ):
    """Plot results from loop_2D_ms, showing the breakdown of
    Bn contributions from currents parallel and perpendicular to the local
    B field.

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    
    # Read pickle file with results from loop_2D_ms
    pklname = info['run_name'] + '.ms-2D.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    # plot results
    plots = [None] 
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ at (1,0,0)',
                        [r'$B_N$ due to $\mathbf{j}$', \
                         r'$B_N$ due to $\mathbf{j}_\parallel$', \
                         r'$B_N$ due to $\mathbf{j}_\perp \cdot \hat \phi$', \
                         r'$B_N$ due to $\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi$'], 
                        time_limits, Bn_limits, r'$B_N$ and components')   
                
    plot_NxM_multiy(info['dir_plots'], 'ms', 'parallel-perpendicular-composition', 
                    plots, cols=1, rows=1, plottype = 'line')
    
    return

def loop_2D_gap_iono(XSM, info, reduce, nTheta=30, nPhi=30, nR=30, delta_hr=None):
    """Loop thru data in RIM files to create data for 2D plots showing Bn versus 
    time including the breakdown of contributions from currents parallel and 
    perpendicular to B field components.  This routine examines field aligned 
    currents (gap) and Pedersen and Hall currents (ionosphere) 

    Inputs:
        XSM = cartesian position where magnetic field will be assessed (SM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents for gap integrals

        delta_hr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the pickle file saved
    """

    # Make sure delta_hr is float
    if delta_hr is not None:
        assert( type(delta_hr) == float )
 
    # Time associated with each file
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Setup temporary variables where results from loop will be saved
    B_gap = [None] * n
    B_pedersen = [None] * n
    B_hall = [None] * n
    B_times = [None] * n
    B_index = [None] * n

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate gap and ionosphere dB for 2D... {base}')
    
        # Record time and index for plots
        if delta_hr is None:
            h = time[3]
            m = time[4]
            B_times[i] = h + m/60
            B_index[i] = i
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=delta_hr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            B_times[i] = h + m/60
            B_index[i] = i
        
        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Field Aligned 
        # Currents in the gap region
        B_gap[i], Be, Bd, Bx, By, Bz = \
            calc_gap_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Pedersen and Hall
        # currents in the ionosphere.
        B_pedersen[i], Bep, Bdp, Bxp, Byp, Bzp, B_hall[i], Beh, Bdh, Bxh, Byh, Bzh = \
            calc_iono_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'])

    # Create dataframe from results and save to disk
    dtimes = [datetime(*time) for time in times]

    df = pd.DataFrame( { r'Gap Total': B_gap, 
                        r'Pedersen Total': B_pedersen, 
                        r'Hall Total': B_hall, 
                        r'Time (hr)': B_times },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.gap-iono-2D.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def loop_2D_gap_iono_point(point, info, reduce, nTheta=30, nPhi=30, nR=30, delta_hr=None):
    """Loop thru data in RIM files to create data for 2D plots showing Bn versus 
    time including the breakdown of contributions from currents parallel and 
    perpendicular to B field components.  This routine examines field aligned 
    currents (gap) and Pedersen and Hall currents (ionosphere) 

    Inputs:
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
                    
        info = info on files to be processed, see info = {...} example above
             
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents for gap integrals

        delta_hr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the pickle file saved
    """

    # Make sure delta_hr is float
    if delta_hr is not None:
        assert( type(delta_hr) == float )
 
        # Time associated with each file
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Setup temporary variables where results from loop will be saved
    B_gap = [None] * n
    B_pedersen = [None] * n
    B_hall = [None] * n
    B_times = [None] * n
    B_index = [None] * n

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate gap and ionosphere dB for 2D... {base} at {point}')
    
        # Record time and index for plots
        if delta_hr is None:
            h = time[3]
            m = time[4]
            B_times[i] = h + m/60
            B_index[i] = i
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=delta_hr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            B_times[i] = h + m/60
            B_index[i] = i
        
        # Get the magnetometer position, X, in SM coordinates for compatibility
        # with RIM data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XSM2 = XGEO.convert( 'SM', 'car' )
        XSM = XSM2.data[0]

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Field Aligned 
        # Currents in the gap region
        B_gap[i], Be, Bd, Bx, By, Bz = \
            calc_gap_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Pedersen and Hall
        # currents in the ionosphere.
        B_pedersen[i], Bep, Bdp, Bxp, Byp, Bzp, B_hall[i], Beh, Bdh, Bxh, Byh, Bzh = \
            calc_iono_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'])

    # Create dataframe from results and save to disk
    dtimes = [datetime(*time) for time in times]

    df = pd.DataFrame( { r'Gap Total': B_gap, 
                        r'Pedersen Total': B_pedersen, 
                        r'Hall Total': B_hall, 
                        r'Time (hr)': B_times },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.gap-iono-2D-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def plot_2D_gap_iono( info, time_limits, Bn_limits ):
    """Plot results from loop_2D_gap_iono, showing the breakdown of
    Bn contributions from field aligned currents (gap) and Pedersen and Hall 
    currents (ionosphere) 

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    pklname = info['run_name'] + '.gap-iono-2D.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    plots = [None] 
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Gap Total', r'Pedersen Total', r'Hall Total'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ at (1,0,0)',
                        [r'$B_N$ due to $\mathbf{j}_{Gap \parallel}$', \
                         r'$B_N$ due to $\mathbf{j}_{Pedersen}$', \
                         r'$B_N$ due to $\mathbf{j}_{Hall}$'], 
                        time_limits, Bn_limits, r'$B_N$ and components')   
                
    plot_NxM_multiy(info['dir_plots'], 'gap-iono', 'parallel-perpendicular-composition', 
                    plots, cols=1, rows=1, plottype = 'line')
    
    return

def plot_2D_ms_gap_iono( info, time_limits, Bn_limits ):
    """Plot results from loop_2D_ms and loop_sum_sb_gap_iono, showing the 
    contibutions to Bn from currents parallel and perpendicular to the local B 
    field in the magnetosphere, field aligned currents in the gap region, and
    Pedersen and Hall currents in the ionosphere

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    
    # Read my data
    pklname = info['run_name'] + '.ms-2D.pkl'
    df1 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    pklname = info['run_name'] + '.gap-iono-2D.pkl'
    df2 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    df1.columns =['MS Total','MS Parallel','MS Perpendicular', r'MS Perpendicular $\phi$', \
                  'MS Perpendicular Residual', 'Time (hr)'  ]

    # Plot results
    plots = [None] * 3
    
    plots[0] = plotargs_multiy(df1, r'Time (hr)', 
                        ['MS Parallel', 'MS Perpendicular'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ (nT) at (1,0,0) (SM)',
                        [r'$B_N$ due to $\mathbf{j}_{\parallel}$', r'$B_N$ due to $\mathbf{j}_{\perp}$'], 
                        time_limits, Bn_limits, r'Magnetosphere')   
            
   
    plots[1] = plotargs_multiy(df2, r'Time (hr)', 
                        ['Gap Total'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ (nT) at (1,0,0) (SM)',
                        [r'$B_N$ due to $\mathbf{j}_{\parallel}$'], 
                        time_limits, Bn_limits, r'Gap FAC')   
            
      
    plots[2] = plotargs_multiy(df2, r'Time (hr)', 
                        ['Hall Total', 'Pedersen Total'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ (nT) at (1,0,0) (SM)',
                        [r'$B_N$ due to $\mathbf{j}_{Hall}$', r'$B_N$ due to $\mathbf{j}_{Pedersen}$'], 
                        time_limits, Bn_limits, r'Ionosphere')   
        
  
    plot_NxM_multiy(info['dir_plots'], 'all', 'parallel-perpendicular-composition', 
                    plots, cols=3, rows=1, plottype = 'line')
    
    return




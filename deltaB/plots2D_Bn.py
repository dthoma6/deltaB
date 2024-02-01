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
from spacepy.time import Ticktock

from deltaB import calc_ms_b_paraperp, \
    calc_gap_b, calc_gap_b_rim, calc_iono_b, \
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

def loop_2D_ms(XSM, info, reduce, deltahr=None, maxcores=20):
    """Loop thru data in BATSRUS files to generate data for 2D plots showing Bn 
    versus time including the breakdown of contributions from currents parallel 
    and perpendicular to local B field.  This routine examines currents in the
    magnetosphere.

    Inputs:
        XSM = cartesian position where magnetic field will be assessed (SM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file saved
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_ms( i, times, deltahr, XSM, info ):
        time = times[i]
        
        # We need the filepath for the BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB for 2D... {base}')
    
        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    
        # Record time and index for plots
        if deltahr is None:
            h = time[3]
            m = time[4]
            Btime = h + m/60
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            Btime = h + m/60
        
        # Convert XSM to GSM coordinates, which is what calc_ms_b_paraperp needs
        XGSM = SMtoGSM(XSM, time, ctype_in='car', ctype_out='car')
    
        # We want the Bn contributions from the main components of the field - 
        # the complete field, and that due to currents parallel and perpendicular
        # to the local B field (perpphi and perpphires are components of perpendicular)
        Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires = \
                calc_ms_b_paraperp(XGSM, timeISO, df)
                
        return Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires, Btime

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_ms)( p, times, deltahr, XSM, info ) 
                                   for p in range(len(times)))
        
        Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires, Btimes = zip(*results)
        
    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bms = [None] * n
        Bms_parallel = [None] * n
        Bms_perp = [None] * n
        Bms_perpphi = [None] * n
        Bms_perpphires = [None] * n
        Btimes = [None] * n

        for p in range(len(times)):
            Bms[p], Bms_parallel[p], Bms_perp[p], Bms_perpphi[p], \
                Bms_perpphires[p], Btimes[p] =  wrap_ms( p, times, deltahr, XSM, info ) 

    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]

    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    df = pd.DataFrame( { r'Total': Bms, 
                        r'Parallel': Bms_parallel, 
                        r'Perpendicular': Bms_perp, 
                        r'Perpendicular $\phi$': Bms_perpphi, 
                        r'Perpendicular Residual': Bms_perpphires,
                        r'Time (hr)': Btimes,
                        r'Datetime': dtimes,
                        r'Month': dtimes_m,
                        r'Day': dtimes_d,
                        r'Hour': dtimes_hh,
                        r'Minute': dtimes_mm },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.ms-2D.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def loop_2D_ms_point(point, info, reduce, deltahr=None, maxcores=20 ):
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
        
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file saved
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_ms( i, times, deltahr, XGEO, info ):
        time = times[i]
        
        # We need the filepath for the BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere dB for 2D... {base}')

        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])

        # Record time and index for plots
        if deltahr is None:
            h = time[3]
            m = time[4]
            Btime = h + m/60
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time)
            dtime2 = dtime + timedelta(hours=deltahr)
            timeISO = dtime2.isoformat()
            h = dtime2.hour
            m = dtime2.minute
            Btime = h + m/60
                
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # BATSRUS data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM2 = XGEO.convert( 'GSM', 'car' )
        XGSM = XGSM2.data[0]

        # We want the Bn contributions from the main components of the field - 
        # the complete field, and that due to currents parallel and perpendicular
        # to the local B field (perpphi and perpphires are components of perpendicular)
        Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires = \
            calc_ms_b_paraperp(XGSM, timeISO, df)
            
        return Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires, Btime

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
    
    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_ms)( p, times, deltahr, XGEO, info ) 
                                   for p in range(len(times)))
        
        Bms, Bms_parallel, Bms_perp, Bms_perpphi, Bms_perpphires, Btimes = zip(*results)
        
    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bms = [None] * n
        Bms_parallel = [None] * n
        Bms_perp = [None] * n
        Bms_perpphi = [None] * n
        Bms_perpphires = [None] * n
        Btimes = [None] * n

        for p in range(len(times)):
            Bms[p], Bms_parallel[p], Bms_perp[p], Bms_perpphi[p], \
                Bms_perpphires[p], Btimes[p] =  wrap_ms( p, times, deltahr, XGEO, info ) 

    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]

    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    df = pd.DataFrame( { r'Total': Bms, 
                        r'Parallel': Bms_parallel, 
                        r'Perpendicular': Bms_perp, 
                        r'Perpendicular $\phi$': Bms_perpphi, 
                        r'Perpendicular Residual': Bms_perpphires,
                        r'Time (hr)': Btimes,
                        r'Datetime': dtimes,
                        r'Month': dtimes_m,
                        r'Day': dtimes_d,
                        r'Hour': dtimes_hh,
                        r'Minute': dtimes_mm},
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.ms-2D-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def plot_2D_ms( point, info, time_limits, Bn_limits ):
    """Plot results from loop_2D_ms, showing the breakdown of
    Bn contributions from currents parallel and perpendicular to the local
    B field.

    Inputs:
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config.  If None, an XSM
            coordinate was provided, not a point name.
                    
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    
    # Read pickle file with results from loop_2D_ms
    if point is not None:
        pklname = info['run_name'] + '.ms-2D-' + point + '.pkl'
        title = r'$B_N$ at ' + point
    else:
        pklname = info['run_name'] + '.ms-2D.pkl'
        title = r'$B_N$'
    df = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    # plot results
    plots = [None] 
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        title,
                        [r'$B_N$ due to $\mathbf{j}$', \
                         r'$B_N$ due to $\mathbf{j}_\parallel$', \
                         r'$B_N$ due to $\mathbf{j}_\perp \cdot \hat \phi$', \
                         r'$B_N$ due to $\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi$'], 
                        time_limits, Bn_limits, r'$B_N$ and components')   
                
    plot_NxM_multiy(info['dir_plots'], 'ms', 'parallel-perpendicular-composition', 
                    plots, cols=1, rows=1, plottype = 'line')
    
    return
    
def loop_2D_gap_iono(XSM, info, reduce, nTheta=180, nPhi=180, nR=800, deltahr=None, 
                     useRIM=True, maxcores=20):
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

        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        useRIM = Boolean, if False use calc_gap_b, if True use calc_gap_b_rim
            The difference is calc_gap_b makes no assumptions about the RIM file
            while calc_gap_b assumes a structure to the RIM file based on 
            reverse engineering.
            
        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file saved
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_gap_iono( i, times, deltahr, XSM, info, nTheta, nPhi, nR, useRIM ):
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate gap and ionosphere dB for 2D... {base}')
    
        # Record time and index for plots
        if deltahr is None:
            h = time[3]
            m = time[4]
            Btime = h + m/60
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            Btime = h + m/60
        
        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Field Aligned 
        # Currents in the gap region
        if useRIM:
            Bgap, Beg, Bdg, Bxg, Byg, Bzg = \
                calc_gap_b_rim(XSM, filepath, timeISO, info['rCurrents'], \
                               info['rIonosphere'], nR)
        else:
            Bgap, Beg, Bdg, Bxg, Byg, Bzg = \
                calc_gap_b(XSM, filepath, timeISO, info['rCurrents'], \
                           info['rIonosphere'], nTheta, nPhi, nR)

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Pedersen and Hall
        # currents in the ionosphere.
        Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, Bhall, Beh, Bdh, Bxh, Byh, Bzh = \
            calc_iono_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'])

        return Bgap, Beg, Bdg, Bxg, Byg, Bzg, Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, \
            Bhall, Beh, Bdh, Bxh, Byh, Bzh, Btime
    
    # Warn user if nTheta and nPhi are ignored
    if useRIM:
        logging.info("nTheta and nPhi values ignored when useRIM is True")
    else:
        logging.warning("Warning: Depreciated mode, useRIM=True is recommended")
        
    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Time associated with each file
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_gap_iono)( p, times, deltahr, \
                                                    XSM, info, nTheta, nPhi, nR, useRIM )
                                   for p in range(len(times)))
        
        Bgap, Beg, Bdg, Bxg, Byg, Bzg, Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, \
            Bhall, Beh, Bdh, Bxh, Byh, Bzh, Btimes = zip(*results)

    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bgap = [None] * n
        Beg = [None] * n
        Bdg = [None] * n
        Bxg = [None] * n
        Byg = [None] * n
        Bzg = [None] * n
        Bpedersen = [None] * n
        Bep = [None] * n
        Bdp = [None] * n
        Bxp = [None] * n
        Byp = [None] * n
        Bzp = [None] * n
        Bhall = [None] * n
        Beh = [None] * n
        Bdh = [None] * n
        Bxh = [None] * n
        Byh = [None] * n
        Bzh = [None] * n
        
        Btimes = [None] * n

        for p in range(len(times)):
            Bgap[p], Beg[p], Bdg[p], Bxg[p], Byg[p], Bzg[p], Bpedersen[p], \
                Bep[p], Bdp[p], Bxp[p], Byp[p], Bzp[p], \
                Bhall[p], Beh[p], Bdh[p], Bxh[p], Byh[p], Bzh[p], Btimes[p] = \
                wrap_gap_iono( p, times, deltahr, XSM, info, nTheta, nPhi, nR, useRIM )

    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]     

    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    df = pd.DataFrame( { r'Gap Total': Bgap, 
                        r'Pedersen Total': Bpedersen, 
                        r'Hall Total': Bhall, 
                        r'Time (hr)': Btimes,
                        r'Datetime': dtimes,
                        r'Month': dtimes_m,
                        r'Day': dtimes_d,
                        r'Hour': dtimes_hh,
                        r'Minute': dtimes_mm },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.gap-iono-2D.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def loop_2D_gap_iono_point(point, info, reduce, nTheta=180, nPhi=180, nR=800, 
                           deltahr=None, useRIM=True, maxcores=20):
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
            between rIonosphere and rCurrents for gap integrals.  NOTE: nTheta
            and nPhi ignored if useRIM is True.  NOTE: if useRIM is True, nTheta
            and nPhi are ignored.

        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.
            
        useRIM = Boolean, if False use calc_gap_b, if True use calc_gap_b_rim
            The difference is calc_gap_b makes no assumptions about the RIM file
            while calc_gap_b assumes a structure to the RIM file based on 
            reverse engineering.
            
        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file saved
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_gap_iono( i, times, deltahr, XGEO, info, nTheta, nPhi, nR, useRIM ):
        time = times[i]
            
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate gap and ionosphere dB for 2D... {base}')

        # Record time and index for plots
        if deltahr is None:
            h = time[3]
            m = time[4]
            Btime = h + m/60
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            Btime = h + m/60
        
        # Get the magnetometer position, X, in SM coordinates for compatibility
        # with RIM data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XSM2 = XGEO.convert( 'SM', 'car' )
        XSM = XSM2.data[0]

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Field Aligned 
        # Currents in the gap region
        if useRIM:
            Bgap, Beg, Bdg, Bxg, Byg, Bzg = \
                calc_gap_b_rim(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'], nR)
        else:
            Bgap, Beg, Bdg, Bxg, Byg, Bzg = \
                calc_gap_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)

        # Get the B field at the point XSM and timeISO using the RIM data
        # results are in SM coordinates.  This call looks at Pedersen and Hall
        # currents in the ionosphere.
        Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, Bhall, Beh, Bdh, Bxh, Byh, Bzh = \
            calc_iono_b(XSM, filepath, timeISO, info['rCurrents'], info['rIonosphere'])

        return Bgap, Beg, Bdg, Bxg, Byg, Bzg, Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, \
            Bhall, Beh, Bdh, Bxh, Byh, Bzh, Btime

    # Warn user if nTheta and nPhi are ignored
    if useRIM:
        logging.info("nTheta and nPhi values ignored when useRIM is True")
    else:
        logging.warning("Warning: Depreciated mode, useRIM=True is recommended")
        
    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Time associated with each file
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_gap_iono)( p, times, deltahr, \
                                                    XGEO, info, nTheta, nPhi, nR, useRIM )
                                   for p in range(len(times)))
        
        Bgap, Beg, Bdg, Bxg, Byg, Bzg, Bpedersen, Bep, Bdp, Bxp, Byp, Bzp, \
            Bhall, Beh, Bdh, Bxh, Byh, Bzh, Btimes = zip(*results)

    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bgap = [None] * n
        Beg = [None] * n
        Bdg = [None] * n
        Bxg = [None] * n
        Byg = [None] * n
        Bzg = [None] * n
        Bpedersen = [None] * n
        Bep = [None] * n
        Bdp = [None] * n
        Bxp = [None] * n
        Byp = [None] * n
        Bzp = [None] * n
        Bhall = [None] * n
        Beh = [None] * n
        Bdh = [None] * n
        Bxh = [None] * n
        Byh = [None] * n
        Bzh = [None] * n
        
        Btimes = [None] * n

        for p in range(len(times)):
            Bgap[p], Beg[p], Bdg[p], Bxg[p], Byg[p], Bzg[p], Bpedersen[p], \
                Bep[p], Bdp[p], Bxp[p], Byp[p], Bzp[p], \
                Bhall[p], Beh[p], Bdh[p], Bxh[p], Byh[p], Bzh[p], Btimes[p] = \
                wrap_gap_iono( p, times, deltahr, XGEO, info, nTheta, nPhi, nR, useRIM )
                
    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]
        
    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    df = pd.DataFrame( { r'Gap Total': Bgap, 
                        r'Pedersen Total': Bpedersen, 
                        r'Hall Total': Bhall, 
                        r'Time (hr)': Btimes,
                        r'Datetime': dtimes,
                        r'Month': dtimes_m,
                        r'Day': dtimes_d,
                        r'Hour': dtimes_hh,
                        r'Minute': dtimes_mm },
                      index=dtimes)

    create_directory(info['dir_derived'], '2D')
    pklname = info['run_name'] + '.gap-iono-2D-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    return

def plot_2D_gap_iono( point, info, time_limits, Bn_limits ):
    """Plot results from loop_2D_gap_iono, showing the breakdown of
    Bn contributions from field aligned currents (gap) and Pedersen and Hall 
    currents (ionosphere) 

    Inputs:
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config.  If None, an XSM 
            coordinate was provided.
                    
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    if point is not None:
        pklname = info['run_name'] + '.gap-iono-2D-' + point + '.pkl'
        title = r'$B_N$ (nT) at ' + point
    else:
        pklname = info['run_name'] + '.gap-iono-2D.pkl'
        title = r'$B_N$ (nT)'
    df = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    plots = [None] 
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Gap Total', r'Pedersen Total', r'Hall Total'], 
                        False, False, 
                        r'Time (hr)',
                        title,
                        [r'$B_N$ due to $\mathbf{j}_{Gap \parallel}$', \
                         r'$B_N$ due to $\mathbf{j}_{Pedersen}$', \
                         r'$B_N$ due to $\mathbf{j}_{Hall}$'], 
                        time_limits, Bn_limits, r'$B_N$ and components')   
                
    plot_NxM_multiy(info['dir_plots'], 'gap-iono', 'parallel-perpendicular-composition', 
                    plots, cols=1, rows=1, plottype = 'line')
    
    return

def plot_2D_ms_gap_iono( point, info, time_limits, Bn_limits ):
    """Plot results from loop_2D_ms and loop_sum_sb_gap_iono, showing the 
    contibutions to Bn from currents parallel and perpendicular to the local B 
    field in the magnetosphere, field aligned currents in the gap region, and
    Pedersen and Hall currents in the ionosphere

    Inputs:
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config.  If None, an XSM
            coordinate was supplied, not a point name.
                    
        info = info on files to be processed, see info = {...} example above
             
        time_limits = time axis limits, e.g., [4,16]
        
        Bn_limits = Bn contributions axis limits, e.g., [-1200,400]
        
    Outputs:
        None - other than the plot file saved
    """
    
    # Read my data
    if point is not None:
        pklname1 = info['run_name'] + '.ms-2D-' + point + '.pkl' 
        pklname2 = info['run_name'] + '.gap-iono-2D-' + point + '.pkl'
        title = r'$B_N$ (nT) at ' + point
    else:
        pklname1 = info['run_name'] + '.ms-2D.pkl'
        pklname2 = info['run_name'] + '.gap-iono-2D.pkl'
        title = r'$B_N$ (nT)'
    
    df1 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname1) )
    df2 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname2) )
        

    df1.columns =['MS Total','MS Parallel','MS Perpendicular', r'MS Perpendicular $\phi$', \
                  'MS Perpendicular Residual', 'Time (hr)', 'Datetime', 'Month', \
                  'Day', 'Hour', 'Minute' ]

    # Plot results
    plots = [None] * 3
    
    plots[0] = plotargs_multiy(df1, r'Time (hr)', 
                        ['MS Parallel', 'MS Perpendicular'], 
                        False, False, 
                        r'Time (hr)',
                        title,
                        [r'$B_N$ due to $\mathbf{j}_{\parallel}$', r'$B_N$ due to $\mathbf{j}_{\perp}$'], 
                        time_limits, Bn_limits, r'Magnetosphere')   
            
   
    plots[1] = plotargs_multiy(df2, r'Time (hr)', 
                        ['Gap Total'], 
                        False, False, 
                        r'Time (hr)',
                        title,
                        [r'$B_N$ due to $\mathbf{j}_{\parallel}$'], 
                        time_limits, Bn_limits, r'Gap FAC')   
            
      
    plots[2] = plotargs_multiy(df2, r'Time (hr)', 
                        ['Hall Total', 'Pedersen Total'], 
                        False, False, 
                        r'Time (hr)',
                        title,
                        [r'$B_N$ due to $\mathbf{j}_{Hall}$', r'$B_N$ due to $\mathbf{j}_{Pedersen}$'], 
                        time_limits, Bn_limits, r'Ionosphere')   
        
  
    plot_NxM_multiy(info['dir_plots'], 'all', 'parallel-perpendicular-composition', 
                    plots, cols=3, rows=1, plottype = 'line')
    
    return




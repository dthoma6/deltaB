#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:14:44 2024

@author: Dean Thomas
"""

from numba import njit
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy.time import Ticktock
import os.path

from deltaB.util import create_directory, date_timeISO
from deltaB.coordinates import GSMtoSM, iso2ints, get_NED_components
from deltaB.BATSRUS_dataframe import get_batsrus_data_from_cdf
from deltaB.BATSRUS_divBint_b import BATSRUS_divBint_b
from deltaB.OpenGGCM_dataframe import get_openggcm_data_from_cdf
from deltaB.OpenGGCM_divBint_b import OpenGGCM_divBint_b
from deltaB.LFM_dataframe import get_lfm_data_from_cdf
from deltaB.LFM_divBint_b import LFM_divBint_b
  
def calc_ms_divBint_b(XGSM, timeISO, mhd):
    """Process data in MHD file to calculate the delta B at point XGSM.
    Helmholtz decomposition theorem used to convert Biot-Savart Law to a 
    surface integral used for calculation.  We will integrate across the outer
    boundary of the MHD grid.  
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in MHD file
              
        mhd = MHD data from BATSRUS, OpenGGCM, etc.
                
    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(r'Calculate magnetosphere divB integral dB...')

    # We need the time to switch from GSM to SM coordinates
    time = iso2ints( timeISO )

    # Results in GSM coordinates
    BGSM = np.zeros(3)

    # Do volume integral
    if mhd.model == 'BATSRUS':
        BGSM = BATSRUS_divBint_b(XGSM, timeISO, mhd)
    elif mhd.model == 'OpenGGCM':
        BGSM = OpenGGCM_divBint_b(XGSM, timeISO, mhd)
    elif mhd.model == 'LFM':
        BGSM = LFM_divBint_b(XGSM, timeISO, mhd)

    # Convert to SM coordinates        
    B = np.zeros(3)
    B = GSMtoSM( BGSM, time, ctype_in='car', ctype_out='car')
    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')
    
    # Get north, east, down components
    Bn, Be, Bd = get_NED_components( B, XSM )
     
    return Bn, Be, Bd, B[0], B[1], B[2]      


# Example info.  Info is used below in call to loop_ms_b
# info = {
#         "model": "SWMF",
#         "run_name": "SWPC_SWMF_052811_2",
#         "rCurrents": 4.0,
#         "file_type": "cdf",
#         "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
#         "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
#         "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
#         "deltaB_files": {
#             "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
#         }
# }

def loop_ms_divBint_b(info, point, reduce, deltahr=None, maxcores=20, deltaBlist=False):
    """Use surface integral at outer boundary from Helmholtz Decomposition Theorem 
    in calc_ms_surfint_outer_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Surface integral uses 
    magnetosphere current density as defined in MHD files

    Inputs:
        info = information on MHD data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list
            
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.
            
        maxcores = for parallel processing, the maximum number of cores to use
        
        deltaBlist = Boolean.  False use magnetpost list of magnetometer sites.
            True use deltaB list of magnetometer sites.
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_ms( i, times, deltahr, XGEO, info ):
        time = times[i]
        
        # We need the filepath for MHD file
        filepath = info['files']['magnetosphere'][times[i]]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere divB integral dB for... {base}')
        
        # We need the ISO time to update the magnetometer position
        # Record time for plots
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
        
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # MHD data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM = XGEO.convert( 'GSM', 'car' )
        X = XGSM.data[0]
    
        # Read in the MHD file 
        if info['model'] == 'SWMF' or info['model'] == 'BATSRUS':
            mhd = get_batsrus_data_from_cdf(filepath,info)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath,info)
        elif info['model'] == 'LFM':
            mhd = get_lfm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')

        # Use Helmholtz decomposition surface integral to calculate magnetic 
        # field, B, at magnetometer position X (GSM).  Store the results, which 
        # are in SM coordinates, and the time
        Bn, Be, Bd, Bx, By, Bz = calc_ms_divBint_b(X, timeISO, mhd)
        
        return Bn, Be, Bd, Bx, By, Bz, Btime

    # Verify input parameters
    assert isinstance(point, str)

    # Make sure delta_hr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Get times for BATSRUS files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types    
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]

    # We need the magnetometer coordinates at point.  Either look it up
    # in the magnetopost list or in deltaB list
    from spacepy import coordinates as coord
    if deltaBlist == False:
        # Get the magnetometer location using magnetopost list
        from magnetopost.config import defined_magnetometers
        pointX = defined_magnetometers[point]
        XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)
    else:
        # Get the magnetometer location from the deltaB list
        from deltaB.magnetometers import specified_magnetometers
        pointX = specified_magnetometers[point]
        XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)
        
    # Loop through the files using parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, len(times), maxcores)
    logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
    results = Parallel(n_jobs=num_cores)(delayed(wrap_ms)( p, times, deltahr, XGEO, info ) 
                               for p in range(len(times)))
    
    Bn, Be, Bd, Bx, By, Bz, Btimes = zip(*results)
        
    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]
        
    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bn': Bn, 'Be': Be, 'Bd': Bd,
                'Bx': Bx, 'By': By, 'Bz': Bz, 
                r'Time (hr)': Btimes, r'Datetime': dtimes,
                r'Month': dtimes_m, r'Day': dtimes_d,
                r'Hour': dtimes_hh, r'Minute': dtimes_mm}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_divB_msph_b-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    

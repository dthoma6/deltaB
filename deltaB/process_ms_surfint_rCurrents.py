#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:14:44 2024

@author: Dean Thomas
"""

# from numba import njit
# import swmfio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy.time import Ticktock
import os.path

from deltaB.util import create_directory, date_timeISO
from deltaB.coordinates import GSMtoSM, iso2ints, get_NED_components
from deltaB.BATSRUS_dataframe import get_batsrus_data_from_cdf
from deltaB.BATSRUS_surfint_rCurrents_b import BATSRUS_surfint_rCurrents_b
from deltaB.OpenGGCM_dataframe import get_openggcm_data_from_cdf
from deltaB.OpenGGCM_surfint_rCurrents_b import OpenGGCM_surfint_rCurrents_b

def calc_ms_surfint_rCurrents_b(XGSM, timeISO, mhd, nTheta=180, nPhi=180):
    """Process data in BATSRUS file to calculate the delta B at point XGSM.
    Helmholtz decomposition theorem used to convert Biot-Savart Law to a 
    surface integral at rCurrents used for calculation.  We will integrate
    across the surface of a sphere at rCurrents, the boundary between the
    MHD calculation in the magnetosphere and the gap region.  
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        mhd = MHD data from BATSRUS, OpenGGCM, etc.
        
        timeISO = ISO time for data in MHD file
              
        nTheta, nPhi = number of steps in numerical integration over theta
            and phi

    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(r'Calculate magnetosphere rCurrents surface integral dB...')

    # We need the time to switch from GSM to SM coordinates
    time = iso2ints( timeISO )

    # Results in GSM coordinates
    BGSM = np.zeros(3)
    BirrGSM = np.zeros(3)
    BsolGSM = np.zeros(3)

    # Do surface integral
    if mhd.model == 'BATSRUS':
        BGSM, BirrGSM, BsolGSM = BATSRUS_surfint_rCurrents_b(XGSM, timeISO, mhd, 
                                                         nTheta, nPhi)
    elif mhd.model == 'OpenGGCM':
        BGSM, BirrGSM, BsolGSM = OpenGGCM_surfint_rCurrents_b(XGSM, timeISO, mhd, 
                                                          nTheta, nPhi)

    # Convert to SM coordinates        
    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')

    B = np.zeros(3)
    Birr = np.zeros(3)
    Bsol = np.zeros(3)

    B = GSMtoSM( BGSM, time, ctype_in='car', ctype_out='car')
    Birr = GSMtoSM( BirrGSM, time, ctype_in='car', ctype_out='car')
    Bsol = GSMtoSM( BsolGSM, time, ctype_in='car', ctype_out='car')
    
    Bn, Be, Bd = get_NED_components( B, XSM )
    Birrn, Birre, Birrd = get_NED_components( Birr, XSM )
    Bsoln, Bsole, Bsold = get_NED_components( Bsol, XSM )
    
    return Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, B[0], B[1], B[2]      


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

def loop_ms_surfint_rCurrents_b(info, point, reduce, nTheta=180, nPhi=180, 
                      deltahr=None, maxcores=20, deltaBlist=False):
    """Use surface integral at rCurrents from Helmholtz Decomposition Theorem 
    in calc_ms_surfint_rCurrents_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Surface integral uses 
    magnetosphere current density as defined in MHD files

    Inputs:
        info = information on MHD data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list
            
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        nTheta, nPhi = number of steps in numerical integration over theta
            and phi

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
    def wrap_ms( i, times, deltahr, XGEO, info, nTheta, nPhi ):
        time = times[i]
        
        # We need the filepath for MHD file
        filepath = info['files']['magnetosphere'][times[i]]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere rCurrents surface integral dB for... {base}')
        
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
            mhd = get_batsrus_data_from_cdf(filepath)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')
     
        # Use Helmholtz decomposition surface integral to calculate magnetic 
        # field, B, at magnetometer position X (GSM).  Store the results, which 
        # are in SM coordinates, and the time
        Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
                Bx, By, Bz = calc_ms_surfint_rCurrents_b(X, timeISO, mhd, 
                                               nTheta, nPhi)
        
        return Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
                Bx, By, Bz, Btime

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
    results = Parallel(n_jobs=num_cores)(delayed(wrap_ms)( p, times, deltahr, 
                                                          XGEO, info, nTheta, nPhi ) 
                               for p in range(len(times)))
    
    Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
            Bx, By, Bz, Btimes = zip(*results)
        
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
                'Birrn': Birrn, 'Birre': Birre, 'Birrd': Birrd, 
                'Bsoln': Bsoln, 'Bsole': Bsole, 'Bsold': Bsold, 
                'Bx': Bx, 'By': By, 'Bz': Bz, 
                r'Time (hr)': Btimes, r'Datetime': dtimes,
                r'Month': dtimes_m, r'Day': dtimes_d,
                r'Hour': dtimes_hh, r'Minute': dtimes_mm}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_si_msph_rCurrents-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    

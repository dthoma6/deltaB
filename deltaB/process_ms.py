#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 13:35:29 2023

@author: Dean Thomas
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import os.path

from deltaB.BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_cumulative_sum_dataframe
from deltaB.util import ned
from deltaB.util import create_directory

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

def calc_ms_b(X, filepath, timeISO, rCurrents):
    """Use Biot-Savart to determine the magnetic field (in North-East-Down 
    coordinates) at point X.  Biot-Savart caclculation uses magnetosphere 
    current density.  Bio-Savart integration from rCurrents to max range of
    BATSRUS grid.

    Inputs:
        X = cartesian position (GSM) where magnetic field will be measured
        
        filepath = path to BATSRUS file
        
        timeISO = time (in ISO format) for BATSRUS file
 
        rCurrents = range from earth center below which BATSRUS results are not
            valid
        
     Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X
            
        Bx, By, Bz = cumulative sum of dB data in x-y-z GSM coordinates
    """

    logging.info(f'Calculate magnetosphere dB... {os.path.basename(filepath)}')
    
    df = convert_BATSRUS_to_dataframe(filepath, rCurrents)    
    df = create_deltaB_rCurrents_dataframe(df, X)
    df = create_cumulative_sum_dataframe(df)
    
    n_geo, e_geo, d_geo = ned(timeISO, X, 'GSM')
    
    Bn = df['dBxSum'].iloc[-1] * n_geo[0] + \
        df['dBySum'].iloc[-1] * n_geo[1] + \
        df['dBzSum'].iloc[-1] * n_geo[2]

    Be = df['dBxSum'].iloc[-1] * e_geo[0] + \
        df['dBySum'].iloc[-1] * e_geo[1] + \
        df['dBzSum'].iloc[-1] * e_geo[2]

    Bd = df['dBxSum'].iloc[-1] * d_geo[0] + \
        df['dBySum'].iloc[-1] * d_geo[1] + \
        df['dBzSum'].iloc[-1] * d_geo[2]
        
    return Bn, Be, Bd, df['dBxSum'].iloc[-1], df['dBySum'].iloc[-1], df['dBzSum'].iloc[-1]

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

def loop_ms_b(info, point, reduce):
    """Use Biot-Savart in calc_ms_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Biot-Savart caclculation 
    uses magnetosphere current density as defined in BATSRUS files

    Inputs:
        info = information on BATSRUS data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
            
        reduce = Boolean, do we skip files to save time
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    import os.path
    
    assert isinstance(point, str)

    # Get a list of BATSRUS files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types
    from magnetopost import util as util
    util.setup(info)
    
    times = list(info['files']['magnetosphere'].keys())
    if reduce:
        times = times[0:len(times):15]
    n = len(times)

    # Prepare storage of variables
    Bn = np.zeros(n)
    Be = np.zeros(n)
    Bd = np.zeros(n)
    Bx = np.zeros(n)
    By = np.zeros(n)
    Bz = np.zeros(n)
    
    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype)
    
    # Loop through each BATSRUS file, storing the results along the way
    for i in range(n):
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[i]]
        
        # We need the ISO time to update the magnetometer position
        timeISO = str(times[i][0]) + '-' + str(times[i][1]).zfill(2) + '-' + str(times[i][2]).zfill(2) + 'T' + \
            str(times[i][3]).zfill(2) +':' + str(times[i][4]).zfill(2) + ':' + str(times[i][5]).zfill(2)
        
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # BATSRUS data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM = XGEO.convert( 'GSM', 'car' )
        # X = [XGSM.x[0], XGSM.y[0], XGSM.z[0]]
        X = XGSM.data[0]
            
        # Use Biot-Savart to calculate magnetic field, B, at magnetometer position
        # XGSM.  Store the results and the time
        Bn[i], Be[i], Bd[i], Bx[i], By[i], Bz[i] = calc_ms_b(X, filepath, timeISO, info['rCurrents'])
    
    dtimes = [datetime(*time) for time in times]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bn': Bn, 'Be': Be, 'Bd': Bd,
                        'Bx': Bx, 'By': By, 'Bz': Bz}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_bs_msph-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
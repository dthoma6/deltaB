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
    create_cumulative_sum_dataframe, \
    create_deltaB_spherical_dataframe, \
    create_deltaB_rCurrents_spherical_dataframe, \
    create_cumulative_sum_spherical_dataframe
from deltaB.util import create_directory, get_NED_components, date_timeISO
from deltaB.coordinates import GSMtoSM, iso2ints

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
            provides total B at point X in SM coordinates
            
        Bx, By, Bz = cumulative sum of dB data in x-y-z SM coordinates
    """

    logging.info(f'Calculate magnetosphere dB... {os.path.basename(filepath)}')
    
    df = convert_BATSRUS_to_dataframe(filepath, rCurrents)    
    df = create_deltaB_rCurrents_dataframe(df, X)
    df = create_cumulative_sum_dataframe(df)
       
    # We need to switch from GSM to SM coordinates
    B = [df['dBxSum'].iloc[-1], df['dBySum'].iloc[-1], df['dBzSum'].iloc[-1]]
    time = iso2ints( timeISO )
    BSM = GSMtoSM(B, time, ctype_in='car', ctype_out='car')
    XSM = GSMtoSM(X, time, ctype_in='car', ctype_out='car')
    
    Bn, Be, Bd = get_NED_components( BSM, XSM )
        
    return Bn, Be, Bd, B[0], B[1], B[2]

def calc_ms_b_paraperp(XGSM, timeISO, df):
    """Use Biot-Savart to determine the magnetic field (in North-East-Down 
    coordinates) at point XGSM.  Biot-Savart caclculation uses magnetosphere 
    current density.  Bio-Savart integration from rCurrents to max range of
    BATSRUS grid.  This routine adds a breakdown of the B field contributions
    from currents parallel and perpendicular to the local B field

    Inputs:
        X = cartesian position where magnetic field will be measured (GSM coordinates)
        
        timeISO = time in ISO format -> '2002-02-25T12:20:30'
        
        df = dataframe with BATRUS data.  df typically generated by a call to
            convert_BATSRUS_to_dataframe

      Outputs:
        Bn = north component of total B field at point X (SM coordinates)
        
        Bparan = north component of B field at point X due to currents
            parallel to B field (SM coordinates)
            
        Bperpn = north component of B field at point X due to currents
            penpendicular to B field (SM coordinates)
            
        Bperpphin = north component of B field at point X due to currents
            perpendicular to B field and in phi-hat direction 
            (j_perpendicular dot phi-hat) (SM coordinates)
            
        Bperpresn = = north component of B field at point X due to 
            residual currents perpendicular to B field (j_perpendicular minus 
            j_perpendicular dot phi-hat) (SM coordinates)
    """

    logging.info('Calculate magnetosphere dB... ')

    df1 = create_deltaB_rCurrents_dataframe(df, XGSM)
    df1 = create_deltaB_spherical_dataframe( df1 )
    df1 = create_deltaB_rCurrents_spherical_dataframe( df1, XGSM )

    logging.info('Calculate cumulative sums...')

    df1 = create_cumulative_sum_dataframe(df1)
    df1 = create_cumulative_sum_spherical_dataframe( df1 )

    # We need the time to switch from GSM to SM coordinates
    time = iso2ints( timeISO )
    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')

    B = [df1['dBxSum'].iloc[-1], \
         df1['dBySum'].iloc[-1], \
         df1['dBzSum'].iloc[-1]]
    BSM = GSMtoSM(B, time, ctype_in='car', ctype_out='car')
    Bn, Be, Bd = get_NED_components(BSM, XSM)     

    Bpara = [df1['dBparallelxSum'].iloc[-1], \
             df1['dBparallelySum'].iloc[-1], \
             df1['dBparallelzSum'].iloc[-1]]
    BparaSM = GSMtoSM(Bpara, time, ctype_in='car', ctype_out='car')
    Bparan, Bparae, Bparad = get_NED_components(BparaSM, XSM)     

    Bperp = [df1['dBperpendicularxSum'].iloc[-1], \
             df1['dBperpendicularySum'].iloc[-1], \
             df1['dBperpendicularzSum'].iloc[-1]]
    BperpSM = GSMtoSM(Bperp, time, ctype_in='car', ctype_out='car')
    Bperpn, Bperpe, Bperpd = get_NED_components(BperpSM, XSM)     

    Bperpphi = [df1['dBperpendicularphixSum'].iloc[-1], \
                df1['dBperpendicularphiySum'].iloc[-1], \
                df1['dBperpendicularphizSum'].iloc[-1]]
    BperpphiSM = GSMtoSM(Bperpphi, time, ctype_in='car', ctype_out='car')
    Bperpphin, Bperpphie, Bperpphid = get_NED_components(BperpphiSM, XSM)     

    Bperpres = [df1['dBperpendicularphiresxSum'].iloc[-1], \
                df1['dBperpendicularphiresySum'].iloc[-1], \
                df1['dBperpendicularphireszSum'].iloc[-1]]
    BperpresSM = GSMtoSM(Bperpres, time, ctype_in='car', ctype_out='car')
    Bperpresn, Bperprese, Bperpresd = get_NED_components(BperpresSM, XSM)     

    return Bn, Bparan, Bperpn, Bperpphin, Bperpresn

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
            
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
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
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
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
        timeISO = date_timeISO( times[i] )
        
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # BATSRUS data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM = XGEO.convert( 'GSM', 'car' )
        X = XGSM.data[0]
            
        # Use Biot-Savart to calculate magnetic field, B, at magnetometer position
        # XGSM.  Store the results, which are in SM coordinates, and the time
        Bn[i], Be[i], Bd[i], Bx[i], By[i], Bz[i] = calc_ms_b(X, filepath, timeISO, info['rCurrents'])
    
    dtimes = [datetime(*time) for time in times]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bn': Bn, 'Be': Be, 'Bd': Bd,
                        'Bx': Bx, 'By': By, 'Bz': Bz}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_bs_msph-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
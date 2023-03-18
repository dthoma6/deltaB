#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:10:35 2023

@author: Dean Thomas
"""

import logging
import numpy as np
import swmfio
import pandas as pd
from datetime import datetime
import os.path

# from deltaB.coordinates import get_NED_vector_components
from deltaB.util import create_directory, get_NED_components, date_timeISO

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

def calc_iono_b(XSM, filepath, timeISO, rCurrents, rIonosphere, nTheta, nPhi, nR):
    """Process data in RIM file to calculate the delta B at point XSM as
    determined by the Pedersen and Hall currents in the ionosphere.  
    Biot-Savart Law is used for calculation.  We will integrate across all 
    Pedersen and Hall currents in the ionosphere at range rIonosphere from earth
    origin.
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        filepath = path to RIM file
        
        timeISO = ISO time for data in RIM file
              
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents
            
    Outputs:
        Bnp, Bep, Bdp = cumulative sum of Pedersen dB data in north-east-down 
            coordinates, provides total BPedersen at point X (in SM coordinates)

        bSMp = total B due to Pedersen currents (in SM coordinates)
        
        Bnh, Beh, Bdh = cumulative sum of Hall dB data in north-east-down coordinates,
            provides total BHall at point X (in SM coordinates)

        bSMh = total B due to Hall currents (in SM coordinates)
        
    """

    logging.info(f'Calculate ionosphere dB... {os.path.basename(filepath)}')

    # Read RIM file
    # swmfio.logger.setLevel(logging.INFO)
    data_arr, var_dict, units = swmfio.read_rim(filepath)
    assert(data_arr.shape[0] != 0)

    df = pd.DataFrame()

    df['x'] = data_arr[var_dict['X']][:]
    df['y'] = data_arr[var_dict['Y']][:]
    df['z'] = data_arr[var_dict['Z']][:]
    
    df['jx'] = data_arr[var_dict['Jx']][:]
    df['jy'] = data_arr[var_dict['Jy']][:]
    df['jz'] = data_arr[var_dict['Jz']][:]

    df['Ex'] = data_arr[var_dict['Ex']][:]
    df['Ey'] = data_arr[var_dict['Ey']][:]
    df['Ez'] = data_arr[var_dict['Ez']][:]

    df['sigmaH'] = data_arr[var_dict['SigmaH']][:]
    df['sigmaP'] = data_arr[var_dict['SigmaP']][:]

    df['measure'] = data_arr[var_dict['measure']][:]
    
    # Get radius center of earth to each x,y,z point
    df['r0'] = np.sqrt( df['x']**2 + df['y']**2 + df['z']**2 )

    # Calculate earth's magnetic field in cartesian coordinates using a 
    # simple dipole model
    #
    # https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
    #
    # As noted in Lotko, magnetic perturbations in the ionosphere 
    # and the low-altitude magnetosphere are much smaller than 
    # the geomagnetic field B0.  So we can use the simple dipole field.
    B0 = 3.12e+4 # Changed to nT units             
    df['Bx'] = - 3 * B0 * df['x'] * df['z'] / df['r0']**5
    df['By'] = - 3 * B0 * df['y'] * df['z'] / df['r0']**5
    df['Bz'] = - B0 * ( 3 * df['z']**2 - df['r0']**2 ) / df['r0']**5
    df['Bmag'] = np.sqrt( df['Bx']**2 + df['By']**2 + df['Bz']**2 )
    df['Bxhat'] = df['Bx'] / df['Bmag']
    df['Byhat'] = df['By'] / df['Bmag']
    df['Bzhat'] = df['Bz'] / df['Bmag']
    
    # Get current density for Pedersen current
    df['Kxp'] = df['Ex'] * df['sigmaP']
    df['Kyp'] = df['Ey'] * df['sigmaP']
    df['Kzp'] = df['Ez'] * df['sigmaP']

    # Get current density for Hall current
    df['Kxh'] = ( df['Byhat']*df['Ez'] - df['Bzhat']*df['Ey'] ) * df['sigmaH']
    df['Kyh'] = ( df['Bzhat']*df['Ex'] - df['Bxhat']*df['Ez'] ) * df['sigmaH']
    df['Kzh'] = ( df['Bxhat']*df['Ey'] - df['Byhat']*df['Ex'] ) * df['sigmaH']

    # Determine range from each cell to point XSM
    df['r'] = ((XSM[0]-df['x'])**2+(XSM[1]-df['y'])**2+(XSM[2]-df['z'])**2)**(1/2)
    
    # Factor needed in delta B calculations below
    # phys['mu0']*phys['Siemens']*phys['mV']/phys['m']/4/np.pi -> 0.1
    df['factor'] = 0.1*df['measure']/df['r']**3
    
    # delta B due to Pedersen currents in each cell
    df['dBxp'] = df['factor']*(df['Kyp']*(XSM[2]-df['z']) - df['Kzp']*(XSM[1]-df['y']))
    df['dByp'] = df['factor']*(df['Kzp']*(XSM[0]-df['x']) - df['Kxp']*(XSM[2]-df['z']))
    df['dBzp'] = df['factor']*(df['Kxp']*(XSM[1]-df['y']) - df['Kyp']*(XSM[0]-df['x']))
    
    # Cumulative Pedersen delta B
    df['dBxpSum'] = df['dBxp'].cumsum()
    df['dBypSum'] = df['dByp'].cumsum()
    df['dBzpSum'] = df['dBzp'].cumsum()

    # delta B due to Hall currents in each cell
    df['dBxh'] = df['factor']*(df['Kyh']*(XSM[2]-df['z']) - df['Kzh']*(XSM[1]-df['y']))
    df['dByh'] = df['factor']*(df['Kzh']*(XSM[0]-df['x']) - df['Kxh']*(XSM[2]-df['z']))
    df['dBzh'] = df['factor']*(df['Kxh']*(XSM[1]-df['y']) - df['Kyh']*(XSM[0]-df['x']))

    # Cumulative Hall delta B
    df['dBxhSum'] = df['dBxh'].cumsum()
    df['dByhSum'] = df['dByh'].cumsum()
    df['dBzhSum'] = df['dBzh'].cumsum()
    
    bSMp = [df['dBxpSum'].iloc[-1], df['dBypSum'].iloc[-1], df['dBzpSum'].iloc[-1]]
    Bnp, Bep, Bdp = get_NED_components( bSMp, XSM )
    
    # n_geo, e_geo, d_geo = ned(timeISO, XSM, 'SM')
    
    # Bnp = np.dot( bSMp, n_geo )
    # Bep = np.dot( bSMp, e_geo )
    # Bdp = np.dot( bSMp, d_geo )
    
    bSMh = [df['dBxhSum'].iloc[-1], df['dByhSum'].iloc[-1], df['dBzhSum'].iloc[-1]]
    Bnh, Beh, Bdh = get_NED_components( bSMh, XSM )

    # Bnh = np.dot( bSMh, n_geo )
    # Beh = np.dot( bSMh, e_geo )
    # Bdh = np.dot( bSMh, d_geo )
        
    return Bnp, Bep, Bdp, bSMp[0], bSMp[1], bSMp[2], Bnh, Beh, Bdh, bSMh[0], bSMh[1], bSMh[2]   

# Example info.  Info is used below in call to loop_ms_b
# info = {
#         "model": "SWMF",
#         "run_name": "SWPC_SWMF_052811_2",
#         "rCurrents": 4.0,
#         "rIonosphere": 1.01725,
#         "file_type": "cdf",
#         "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
#         "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
#         "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
#         "deltaB_files": {
#             "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
#         }
# }

def loop_iono_b(info, point, reduce, nTheta, nPhi, nR):
    """Use Biot-Savart in calc_iono_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Biot-Savart caclculation 
    uses ionosphere current density as defined in RIM files

    Inputs:
        info = information on RIM data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
            
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    import os.path
    
    assert isinstance(point, str)

    # Get a list of RIM files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types
    from magnetopost import util as util
    util.setup(info)
    
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Prepare storage of variables
    Bnp = np.zeros(n)
    Bep = np.zeros(n)
    Bdp = np.zeros(n)
    Bxp = np.zeros(n)
    Byp = np.zeros(n)
    Bzp = np.zeros(n)
    
    Bnh = np.zeros(n)
    Beh = np.zeros(n)
    Bdh = np.zeros(n)
    Bxh = np.zeros(n)
    Byh = np.zeros(n)
    Bzh = np.zeros(n)

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype)
    
    # Loop through each RIM file, storing the results along the way
    for i in range(n):
        
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][times[i]]
        
        # We need the ISO time to update the magnetometer position
        timeISO = date_timeISO( times[i] )
        
        # Get the magnetometer position, X, in SM coordinates for compatibility with
        # RIM data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XSM = XGEO.convert( 'SM', 'car' )
        X = XSM.data[0]
            
        # Use Biot-Savart to calculate magnetic field, B, at magnetometer position
        # XSM.  Store the results, which are in SM coordinates, and the time
        Bnp[i], Bep[i], Bdp[i], Bxp[i], Byp[i], Bzp[i], \
                Bnh[i], Beh[i], Bdh[i], Bxh[i], Byh[i], Bzh[i],= calc_iono_b(X, filepath, \
                timeISO, info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)
    
    dtimes = [datetime(*time) for time in times]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bnp': Bnp, 'Bep': Bep, 'Bdp': Bdp,
                        'Bxp': Bxp, 'Byp': Byp, 'Bzp': Bzp,
                        'Bnh': Bnh, 'Beh': Beh, 'Bdh': Bdh,
                        'Bxh': Bxh, 'Byh': Byh, 'Bzh': Bzh}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_bs_iono-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 08:11:30 2022

@author: Dean Thomas
"""
import logging
from copy import deepcopy
import swmfio
import pandas as pd
import numpy as np
import os.path

def convert_BATSRUS_to_dataframe(file, rCurrents, region=None):
    """Process data in BATSRUS file to create dataframe.  
    
    Inputs:
        file = path to BATSRUS file or BATSRUS data from swmfio
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units.  We drop the data inside radius rCurrents
            
        region = if specified, region in which each BATSRUS grid point lies, 
            see deltaB_by_region for region definitions. region is a numpy array
            of length df['x'].  i.e., has same number of points as the BATSRUS 
            grid.  Otherwise, region must be None
            
    Outputs:
        df = dataframe containing data from BATSRUS file plus additional calculated
            parameters
    """

    # Read BATSRUS file
    if isinstance(file, str):
        batsrus = swmfio.read_batsrus(file)
        logging.info(f'Parsing BATSRUS file... {os.path.basename(file)}')
    else:
        batsrus = file
        logging.info(f'Parsing BATSRUS file... {os.path.basename(file.file)}')

    assert(batsrus != None)

    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)

    df = pd.DataFrame()

    df['x'] = batsrus.data_arr[:, var_dict['x']][:]
    df['y'] = batsrus.data_arr[:, var_dict['y']][:]
    df['z'] = batsrus.data_arr[:, var_dict['z']][:]

    df['bx'] = batsrus.data_arr[:, var_dict['bx']][:]
    df['by'] = batsrus.data_arr[:, var_dict['by']][:]
    df['bz'] = batsrus.data_arr[:, var_dict['bz']][:]

    df['jx'] = batsrus.data_arr[:, var_dict['jx']][:]
    df['jy'] = batsrus.data_arr[:, var_dict['jy']][:]
    df['jz'] = batsrus.data_arr[:, var_dict['jz']][:]

    df['ux'] = batsrus.data_arr[:, var_dict['ux']][:]
    df['uy'] = batsrus.data_arr[:, var_dict['uy']][:]
    df['uz'] = batsrus.data_arr[:, var_dict['uz']][:]

    df['p'] = batsrus.data_arr[:, var_dict['p']][:]
    df['rho'] = batsrus.data_arr[:, var_dict['rho']][:]
    df['measure'] = batsrus.data_arr[:, var_dict['measure']][:]

    # Determine magnitude of various vectors
    df['jMag'] = np.sqrt(df['jx']**2 + df['jy']**2 + df['jz']**2)
    df['uMag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)
    df['r0'] = np.sqrt((df['x'])**2+(df['y'])**2+(df['z'])**2)

    if isinstance(region, np.ndarray):
        assert( len(df['x']) == len(region))
        df['region'] = region
    else:
        assert region is None
        
    # We ignore everything inside of rCurrents
    df = df.drop(df[df['r0'] < rCurrents].index)
    
    return df


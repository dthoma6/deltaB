#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 08:11:30 2022

@author: Dean Thomas
"""
import logging
from copy import deepcopy
import swmfio
from os.path import exists
import pandas as pd
import numpy as np

def convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath, rCurrents):
    """Process data in BATSRUS file to create dataframe.  Biot-Savart Law used
    to calculate delta B in each BATSRUS cell.  In addition, current j and 
    magnetic field dB is determined in various coordinate systems (i.e., spherical 
    coordinates and parallel/perpendicular to B field).
    
    Inputs:
        X,Y,Z = position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units
    Outputs:
        df = dataframe containing data from BATSRUS file plus additional calculated
            parameters
            
        title = title to use in plots, which is derived from base (file basename)
    """

    logging.info('Parsing BATSRUS file...')

    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')

    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
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

    # Get the smallest cell (by volume), we will use it to normalize the
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize the magnetic field
    # to the smallest cell.
    minMeasure = df['measure'].min()

    logging.info('Calculating delta B...')

    # Calculate useful quantities
    df['r'] = ((X-df['x'])**2+(Y-df['y'])**2+(Z-df['z'])**2)**(1/2)

    # We ignore everything inside of rCurrents
    # df = df[df['r'] > rCurrents]
    df.drop(df[df['r'] < rCurrents].index)

    ##########################################################
    # Below we calculate the delta B in each cell from the
    # Biot-Savart Law.  We want the final result to be in nT.
    # dB = mu0/(4pi) (j x r)/r^3 dV
    #    = (4pi 10^(-7) [H/m])/(4pi) (10^(-6) [A/m^2]) [Re] [Re^3] / [Re^3]
    # where the fact that J is in microamps/m^2 and distances are in Re
    # is in the BATSRUS documentation.   We take Re = 6372 km = 6.371 10^6 m
    # dB = 6.371 10^(-7) [H/m][A/m^2][m]
    #    = 6.371 10^(-7) [H] [A/m^2]
    #    = 6.371 10^(-7) [kg m^2/(s^2 A^2)] [A/m^2]
    #    = 6.371 10^(-7) [kg / s^2 / A]
    #    = 6.371 10^(-7) [T]
    #    = 6.371 10^2 [nT]
    #    = 637.1 [nT] with distances in Re, j in microamps/m^2
    ##########################################################

    # Determine delta B in each cell
    df['factor'] = 637.1*df['measure']/df['r']**3
    df['dBx'] = df['factor']*(df['jy']*(Z-df['z']) - df['jz']*(Y-df['y']))
    df['dBy'] = df['factor']*(df['jz']*(X-df['x']) - df['jx']*(Z-df['z']))
    df['dBz'] = df['factor']*(df['jx']*(Y-df['y']) - df['jy']*(X-df['x']))

    # Determine magnitude of various vectors
    df['dBmag'] = np.sqrt(df['dBx']**2 + df['dBy']**2 + df['dBz']**2)
    df['jMag'] = np.sqrt(df['jx']**2 + df['jy']**2 + df['jz']**2)
    df['uMag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)

    # Normalize magnetic field, as mentioned above
    df['dBmagNorm'] = df['dBmag'] * minMeasure/df['measure']
    df['dBxNorm'] = np.abs(df['dBx'] * minMeasure/df['measure'])
    df['dByNorm'] = np.abs(df['dBy'] * minMeasure/df['measure'])
    df['dBzNorm'] = np.abs(df['dBz'] * minMeasure/df['measure'])

    logging.info('Transforming j to spherical coordinates...')

    # Transform the currents, j, into spherical coordinates

    # Determine theta and phi of the radius vector from the origin to the
    # center of the cell
    df['theta'] = np.arccos(df['z']/df['r'])
    df['phi'] = np.arctan2(df['y'], df['x'])

    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df['jr'] = df['jx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jz'] * np.cos(df['theta'])

    df['jtheta'] = df['jx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jz'] * np.sin(df['theta'])

    df['jphi'] = - df['jx'] * np.sin(df['phi']) + df['jy'] * np.cos(df['phi'])

    # Similarly, use dot-products to determine dBr, dBtheta, and dBphi
    df['dBr'] = df['dBx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['dBz'] * np.cos(df['theta'])

    df['dBtheta'] = df['dBx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['dBz'] * np.sin(df['theta'])

    df['dBphi'] = - df['dBx'] * \
        np.sin(df['phi']) + df['dBy'] * np.cos(df['phi'])

    # Use dot product with B/BMag to get j parallel to magnetic field
    # To get j perpendicular, use j perpendicular = j - j parallel 
    df['bMag'] = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2)

    df['jparallelMag'] = (df['jx'] * df['bx'] + df['jy']
                          * df['by'] + df['jz'] * df['bz'])/df['bMag']

    df['jparallelx'] = df['jparallelMag'] * df['bx']/df['bMag']
    df['jparallely'] = df['jparallelMag'] * df['by']/df['bMag']
    df['jparallelz'] = df['jparallelMag'] * df['bz']/df['bMag']

    df['jperpendicularMag'] = np.sqrt(df['jMag']**2 - df['jparallelMag']**2)

    df['jperpendicularx'] = df['jx'] - df['jparallelx']
    df['jperpendiculary'] = df['jy'] - df['jparallely']
    df['jperpendicularz'] = df['jz'] - df['jparallelz']
    
    # Convert j perpendicular to spherical coordinates
    df['jperpendicularr'] = df['jperpendicularx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jperpendicularz'] * np.cos(df['theta'])

    df['jperpendiculartheta'] = df['jperpendicularx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jperpendicularz'] * np.sin(df['theta'])

    df['jperpendicularphi'] = - df['jperpendicularx'] * np.sin(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['phi'])

    # Divide j perpendicular into a phi piece and everything else (residual)
    df['jperpendicularphix'] = - df['jperpendicularphi'] * np.sin(df['phi'])
    df['jperpendicularphiy'] =   df['jperpendicularphi'] * np.cos(df['phi'])
    # df['jperpendicularphiz'] = 0

    df['jperpendicularphiresx'] = df['jperpendicularx'] - df['jperpendicularphix']
    df['jperpendicularphiresy'] = df['jperpendiculary'] - df['jperpendicularphiy']
    # df['jperpendicularphiresz'] = df['jperpendicularz']
    
    # Determine delta B using the parallel and perpendicular currents. They
    # should sum to the delta B calculated above for the full current, jx, jy, jz
    df['dBparallelx'] = df['factor'] * \
        (df['jparallely']*(Z-df['z']) - df['jparallelz']*(Y-df['y']))
    df['dBparallely'] = df['factor'] * \
        (df['jparallelz']*(X-df['x']) - df['jparallelx']*(Z-df['z']))
    df['dBparallelz'] = df['factor'] * \
        (df['jparallelx']*(Y-df['y']) - df['jparallely']*(X-df['x']))

    df['dBperpendicularx'] = df['factor'] * \
        (df['jperpendiculary']*(Z-df['z']) - df['jperpendicularz']*(Y-df['y']))
    df['dBperpendiculary'] = df['factor'] * \
        (df['jperpendicularz']*(X-df['x']) - df['jperpendicularx']*(Z-df['z']))
    df['dBperpendicularz'] = df['factor'] * \
        (df['jperpendicularx']*(Y-df['y']) - df['jperpendiculary']*(X-df['x']))

    # Divide delta B from perpendicular currents into two - those along phi and 
    # everything else (residual)
    df['dBperpendicularphix'] =   df['factor']*df['jperpendicularphiy']*(Z-df['z'])
    df['dBperpendicularphiy'] = - df['factor']*df['jperpendicularphix']*(Z-df['z'])
    df['dBperpendicularphiz'] = df['factor']*(df['jperpendicularphix']*(Y-df['y']) - \
                                              df['jperpendicularphiy']*(X-df['x']))
        
    df['dBperpendicularphiresx'] = df['factor']*(df['jperpendicularphiresy']*(Z-df['z']) - \
                                                  df['jperpendicularz']*(Y-df['y']))
    df['dBperpendicularphiresy'] = df['factor']*(df['jperpendicularz']*(X-df['x']) - \
                                                  df['jperpendicularphiresx']*(Z-df['z']))
    df['dBperpendicularphiresz'] = df['factor']*(df['jperpendicularphiresx']*(Y-df['y']) - \
                                                  df['jperpendicularphiresy']*(X-df['x']))

    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1] + ' (hhmmss)'

    return df, title

def create_cumulative_sum_dataframe(df):
    """Convert the dataframe with BATSRUS data to a dataframe that provides a 
    cumulative sum of dB in increasing r. To generate the cumulative sum, we 
    order the cells in terms of range r starting at the earth's center.  We 
    start the sum with a small sphere and vector sum all of the dB contributions 
    inside the sphere.  We then expand the sphere slightly and add to the sum.  
    Repeat until all cells are in the sum.

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
            
    Outputs:
        df_r = dataframe with all the df data plus the cumulative sums
    """

    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_r = deepcopy(df)
    df_r = df_r.sort_values(by='r', ascending=True)
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBSumMag'] = np.sqrt(
        df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)

    # Do cumulative sums on currents parallel and perpendicular to the B field
    df_r['dBparallelxSum'] = df_r['dBparallelx'].cumsum()
    df_r['dBparallelySum'] = df_r['dBparallely'].cumsum()
    df_r['dBparallelzSum'] = df_r['dBparallelz'].cumsum()
    df_r['dBparallelSumMag'] = np.sqrt(df_r['dBparallelxSum']**2
                                        + df_r['dBparallelySum']**2
                                        + df_r['dBparallelzSum']**2)

    df_r['dBperpendicularxSum'] = df_r['dBperpendicularx'].cumsum()
    df_r['dBperpendicularySum'] = df_r['dBperpendiculary'].cumsum()
    df_r['dBperpendicularzSum'] = df_r['dBperpendicularz'].cumsum()
    df_r['dBperpendicularSumMag'] = np.sqrt(df_r['dBperpendicularxSum']**2
                                            + df_r['dBperpendicularySum']**2
                                            + df_r['dBperpendicularzSum']**2)

    df_r['dBperpendicularphixSum'] = df_r['dBperpendicularphix'].cumsum()
    df_r['dBperpendicularphiySum'] = df_r['dBperpendicularphiy'].cumsum()
    df_r['dBperpendicularphizSum'] = df_r['dBperpendicularphiz'].cumsum()

    df_r['dBperpendicularphiresxSum'] = df_r['dBperpendicularphiresx'].cumsum()
    df_r['dBperpendicularphiresySum'] = df_r['dBperpendicularphiresy'].cumsum()
    df_r['dBperpendicularphireszSum'] = df_r['dBperpendicularphiresz'].cumsum()

    return df_r

def create_jrtp_cdf_dataframes(df):
    """Use the dataframe with BATSRUS dataframe to develop jr, jtheta, jphi
    cummulative distribution functions (CDFs).

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities
        
    Outputs:
        cdf_jr, cdf_jtheta, cdf_jphi = jr, jtheta, jphi CDFs.
    """

    # Sort the original data, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_jr = deepcopy(df)
    df_jtheta = deepcopy(df)
    df_jphi = deepcopy(df)
    
    df_jr = df_jr.sort_values(by='jr', ascending=True)
    df_jr['cdfIndex'] = np.arange(1, len(df_jr)+1)/float(len(df_jr))

    df_jtheta = df_jtheta.sort_values(by='jtheta', ascending=True)
    df_jtheta['cdfIndex'] = np.arange(1, len(df_jtheta)+1)/float(len(df_jtheta))

    df_jphi = df_jphi.sort_values(by='jphi', ascending=True)
    df_jphi['cdfIndex'] = np.arange(1, len(df_jphi)+1)/float(len(df_jphi))

    return df_jr, df_jtheta, df_jphi


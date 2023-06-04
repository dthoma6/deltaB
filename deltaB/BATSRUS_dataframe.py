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

def convert_BATSRUS_to_dataframe(file, rCurrents):
    """Process data in BATSRUS file to create dataframe.  
    
    Inputs:
        file = path to BATSRUS file
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units.  We drop the data inside radius rCurrents
    Outputs:
        df = dataframe containing data from BATSRUS file plus additional calculated
            parameters
    """

    logging.info(f'Parsing BATSRUS file... {os.path.basename(file)}')

    # Read BATSRUS file
    # swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(file)
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

    # We ignore everything inside of rCurrents
    df.drop(df[df['r0'] < rCurrents].index)
    
    return df

def create_deltaB_spherical_dataframe(df):
    """Convert the dataframe with BATSRUS data to a dataframe that includes
    spherical coordinates breakdown of current density 
    
    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
                        
    Outputs:
        df = dataframe with all the df data plus spherical coordinates
    """

    logging.info('Creating spherical coordinate BATSRUS dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # and that it hasn't already been converted
    assert 'x' in df.columns and not 'theta' in df.columns
    
    # Transform the currents, j, into spherical coordinates

    # Determine theta and phi of the radius vector from the origin to the
    # center of the cell
    df['theta'] = np.arccos(df['z']/df['r0'])
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

    logging.info('Determining j parallel and perpendicular to B...')

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
    
    return df

def create_deltaB_rCurrents_dataframe(df, X):
    """Convert the dataframe with BATSRUS data to a dataframe that uses 
    Biot-Savart Law to calculate delta B in each BATSRUS cell outside of 
    rCurrents.  Points inside rCurrents were dropped in convert_BATSRUS_to_dataframe
    
    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
            
        X = cartesian position where magnetic field will be measured in GSM
            coordinates
            
    Outputs:
        df_b = dataframe with all the df data plus the delta B quantities
    """

    logging.info('Creating deltaB >rCurrents BATSRUS dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # and that it hasn't already been converted
    assert 'x' in df.columns and not 'factor' in df.columns

    df_b = deepcopy(df)
    
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
    df_b['r'] = ((X[0]-df_b['x'])**2+(X[1]-df_b['y'])**2+(X[2]-df_b['z'])**2)**(1/2)
    df_b['factor'] = 637.1*df_b['measure']/df_b['r']**3
    
    df_b['dBx'] = df_b['factor']*(df_b['jy']*(X[2]-df_b['z']) - df_b['jz']*(X[1]-df_b['y']))
    df_b['dBy'] = df_b['factor']*(df_b['jz']*(X[0]-df_b['x']) - df_b['jx']*(X[2]-df_b['z']))
    df_b['dBz'] = df_b['factor']*(df_b['jx']*(X[1]-df_b['y']) - df_b['jy']*(X[0]-df_b['x']))

    # Determine magnitude of dB
    df_b['dBmag'] = np.sqrt(df_b['dBx']**2 + df_b['dBy']**2 + df_b['dBz']**2)

    # Get the smallest cell (by volume), we will use it to normalize the
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize the magnetic field
    # to the smallest cell.
    minMeasure = df_b['measure'].min()

    # Normalize magnetic field, as mentioned above
    df_b['dBmagNorm'] = df_b['dBmag'] * minMeasure/df_b['measure']
    df_b['dBxNorm'] = np.abs(df_b['dBx'] * minMeasure/df_b['measure'])
    df_b['dByNorm'] = np.abs(df_b['dBy'] * minMeasure/df_b['measure'])
    df_b['dBzNorm'] = np.abs(df_b['dBz'] * minMeasure/df_b['measure'])
    
    return df_b

def create_deltaB_rCurrents_spherical_dataframe(df, X):
    """Convert the dataframe with BATSRUS data to a dataframe that uses 
    Biot-Savart Law to calculate delta dB in spherical coordinates.  delta B
    is determined in each BATSRUS cell outside of rCurrents.  Cells inside of
    rCurrents were deleted when the dataframe was created in 
    convert_BATSRUS_to_dataframe
    
    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
            
        X = cartesian position where magnetic field will be measured in GSM
            coordinates
            
    Outputs:
        df = dataframe with all the df data plus the delta B quantities
    """

    logging.info('Creating deltaB spherical coordinates BATSRUS dataframe...')

    # Verify that BATSRUS dataframe has spherical elements and it hasn't already
    # been converted
    assert 'theta' in df.columns and not 'dBr' in df.columns

    # Use dot-products to determine dBr, dBtheta, and dBphi
    df['dBr'] = df['dBx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['dBz'] * np.cos(df['theta'])

    df['dBtheta'] = df['dBx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['dBz'] * np.sin(df['theta'])

    df['dBphi'] = - df['dBx'] * \
        np.sin(df['phi']) + df['dBy'] * np.cos(df['phi'])

    # Determine delta B using the currents parallel and perpendicular to B. They
    # should sum to the delta B calculated above for the full current, jx, jy, jz
    df['dBparallelx'] = df['factor'] * \
        (df['jparallely']*(X[2]-df['z']) - df['jparallelz']*(X[1]-df['y']))
    df['dBparallely'] = df['factor'] * \
        (df['jparallelz']*(X[0]-df['x']) - df['jparallelx']*(X[2]-df['z']))
    df['dBparallelz'] = df['factor'] * \
        (df['jparallelx']*(X[1]-df['y']) - df['jparallely']*(X[0]-df['x']))

    df['dBperpendicularx'] = df['factor'] * \
        (df['jperpendiculary']*(X[2]-df['z']) - df['jperpendicularz']*(X[1]-df['y']))
    df['dBperpendiculary'] = df['factor'] * \
        (df['jperpendicularz']*(X[0]-df['x']) - df['jperpendicularx']*(X[2]-df['z']))
    df['dBperpendicularz'] = df['factor'] * \
        (df['jperpendicularx']*(X[1]-df['y']) - df['jperpendiculary']*(X[0]-df['x']))

    # Divide delta B from perpendicular currents into two - those along phi and 
    # everything else (residual)
    df['dBperpendicularphix'] =   df['factor']*df['jperpendicularphiy']*(X[2]-df['z'])
    df['dBperpendicularphiy'] = - df['factor']*df['jperpendicularphix']*(X[2]-df['z'])
    df['dBperpendicularphiz'] = df['factor']*(df['jperpendicularphix']*(X[1]-df['y']) - \
                                              df['jperpendicularphiy']*(X[0]-df['x']))
        
    df['dBperpendicularphiresx'] = df['factor']*(df['jperpendicularphiresy']*(X[2]-df['z']) - \
                                                  df['jperpendicularz']*(X[1]-df['y']))
    df['dBperpendicularphiresy'] = df['factor']*(df['jperpendicularz']*(X[0]-df['x']) - \
                                                  df['jperpendicularphiresx']*(X[2]-df['z']))
    df['dBperpendicularphiresz'] = df['factor']*(df['jperpendicularphiresx']*(X[1]-df['y']) - \
                                                  df['jperpendicularphiresy']*(X[0]-df['x']))

    return df

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

    logging.info('Creating cumulative sum BATSRUS dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # and that it hasn't already been converted
    assert 'x' in df.columns and not 'dBxSum' in df.columns

    # Sort the original data by range r, ascending.  Then calculate the total B 
    # based upon summing the vector dB values starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_r = deepcopy(df)
    df_r = df_r.sort_values(by='r0', ascending=True)
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBSumMag'] = np.sqrt(
        df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)

    return df_r

def create_cumulative_sum_spherical_dataframe(df):
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

    logging.info('Creating cumulative sum BATSRUS dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # and that it hasn't already been converted
    assert 'x' in df.columns and not 'dBparallelxSum' in df.columns

    # Sort the original data by range r, ascending.  Then calculate the total B 
    # based upon summing the vector dB values starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_r = deepcopy(df)
    df_r = df_r.sort_values(by='r0', ascending=True)

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

    logging.info('Creating CDF dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # with needed data
    assert 'jr' in df.columns 

    # Sort the j components (jr, jtheta, jphi) in order of value.  If each entry 
    # in the dataframe is equally likely, the arange is the CDF.

    df_jr = deepcopy(df)
    df_jtheta = deepcopy(df)
    df_jphi = deepcopy(df)
    
    df_jr = df_jr.sort_values(by='jr', ascending=True)
    df_jr['cdf'] = np.arange(1, len(df_jr)+1)/float(len(df_jr))

    df_jtheta = df_jtheta.sort_values(by='jtheta', ascending=True)
    df_jtheta['cdf'] = np.arange(1, len(df_jtheta)+1)/float(len(df_jtheta))

    df_jphi = df_jphi.sort_values(by='jphi', ascending=True)
    df_jphi['cdf'] = np.arange(1, len(df_jphi)+1)/float(len(df_jphi))

    return df_jr, df_jtheta, df_jphi

def create_jpp_cdf_dataframes(df):
    """Use the dataframe with BATSRUS dataframe to develop jparallel and
    jperpendicular cummulative distribution functions (CDFs).

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities
        
    Outputs:
        cdf_jparallel, cdf_jperpendicular = jparallel, jperpendicular CDFs.
    """

    logging.info('Creating CDF dataframe...')
    
    # Verify that we have something that looks like a BATSRUS dataframe
    # with needed data
    assert 'jr' in df.columns 

    # Sort the j components (jr, jtheta, jphi) in order of value.  If each entry 
    # in the dataframe is equally likely, the arange is the CDF.

    df_jparallel = deepcopy(df)
    df_jperpendicular = deepcopy(df)
    
    df_jparallel = df_jparallel.sort_values(by='jparallelMag', ascending=True)
    df_jparallel['cdf'] = np.arange(1, len(df_jparallel)+1)/float(len(df_jparallel))

    df_jperpendicular = df_jperpendicular.sort_values(by='jperpendicularMag', ascending=True)
    df_jperpendicular['cdf'] = np.arange(1, len(df_jperpendicular)+1)/float(len(df_jperpendicular))

    return df_jparallel, df_jperpendicular




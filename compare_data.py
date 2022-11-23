#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:39:37 2022

@author: dean
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import exists
import swmfio
import logging

# origin and target define where input data and output plots are stored
origin = '/Volumes/Physics HD v2/divB_simple1/GM/'
target = '/Volumes/Physics HD v2/divB_simple1/plots/'

# names of BATSRUS and Paraview file
base = '3d__mhd_4_e20100320-000000-000'
paraview = 'data4.csv'

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.8,7.2]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 7
plt.rcParams['axes.grid'] = True

# rCurrents define range from earth center below which results are not valid
# measured in Re units
rCurrents = 1.8

# Initialize useful variables
(X,Y,Z) = (1.0, 0.0, 0.0) 

from math import log10, floor
def round_to_6(x):
    """Round x to 6 significant figures. Used below because Paraview data
    has only 6 significant figures while BATSRUS has more.
        
    Inputs:
        x = quantity to be rounded
    Outputs:
        y = rounded quantity
    """
    y = 0
    if(x != 0): 
        y = np.round(x, -int(floor(log10(abs(x))))+5)
    return y

def diff_over_avg( l1, l2 ):
    """Determine the difference between np.arrays l1 and l2, divided by the
    average value of l1 and l2
    
    Inputs:
        l1, l2 = two np.arrays with the same number of entries
    Outputs:
        difference/average
    """
    assert isinstance( l1, np.ndarray )
    assert isinstance( l2, np.ndarray )
    assert( len(l1) == len(l2) )
    
    return 2*(l1 - l2)/(l1 + l2)

def convert_BATSRUS_to_dataframe(base, dirpath = origin):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
        
    Inputs:
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
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
    assert( batsrus != None )
       
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    df = pd.DataFrame()
    
    df['x'] = batsrus.data_arr[:,var_dict['x']][:]
    df['y'] = batsrus.data_arr[:,var_dict['y']][:]
    df['z'] = batsrus.data_arr[:,var_dict['z']][:]
        
    df['bx'] = batsrus.data_arr[:,var_dict['bx']][:]
    df['by'] = batsrus.data_arr[:,var_dict['by']][:]
    df['bz'] = batsrus.data_arr[:,var_dict['bz']][:]
        
    df['jx'] = batsrus.data_arr[:,var_dict['jx']][:]
    df['jy'] = batsrus.data_arr[:,var_dict['jy']][:]
    df['jz'] = batsrus.data_arr[:,var_dict['jz']][:]
        
    df['ux'] = batsrus.data_arr[:,var_dict['ux']][:]
    df['uy'] = batsrus.data_arr[:,var_dict['uy']][:]
    df['uz'] = batsrus.data_arr[:,var_dict['uz']][:]
        
    df['p'] = batsrus.data_arr[:,var_dict['p']][:]
    df['rho'] = batsrus.data_arr[:,var_dict['rho']][:]
    df['measure'] = batsrus.data_arr[:,var_dict['measure']][:]
    
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
    df['dBx'] = df['factor']*( df['jy']*(Z-df['z']) - df['jz']*(Y-df['y']) )
    df['dBy'] = df['factor']*( df['jz']*(X-df['x']) - df['jx']*(Z-df['z']) )
    df['dBz'] = df['factor']*( df['jx']*(Y-df['y']) - df['jy']*(X-df['x']) )

    # Determine magnitude of various vectors
    df['dBmag'] = np.sqrt(df['dBx']**2 + df['dBy']**2 + df['dBz']**2)    
    df['jMag'] = np.sqrt( df['jx']**2 + df['jy']**2 + df['jz']**2 )
    df['uMag'] = np.sqrt( df['ux']**2 + df['uy']**2 + df['uz']**2 )

    # Normalize magnetic field, as mentioned above
    df['dBmagNorm'] = df['dBmag'] * minMeasure/df['measure']
    df['dBxNorm'] = np.abs(df['dBx'] * minMeasure/df['measure'])
    df['dByNorm'] = np.abs(df['dBy'] * minMeasure/df['measure'])
    df['dBzNorm'] = np.abs(df['dBz'] * minMeasure/df['measure'])

    logging.info('Transforming j to spherical coordinates...')
    
    # Transform the currents, j, into spherical coordinates

    # Determine theta and phi of the radius vector from the origin to the 
    # center of the cell
    df['theta'] = np.arccos( df['z']/df['r'] )
    df['phi'] = np.arctan2( df['y'], df['x'] )
    
    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df['jr'] = df['jx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jz'] * np.cos(df['theta'])
        
    df['jtheta'] = df['jx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jz'] * np.sin(df['theta'])
        
    df['jphi-sc'] = - df['jx'] * np.sin(df['phi']) + df['jy'] * np.cos(df['phi'])
    df['jphi-xy'] = (- df['jx'] * df['y'] + df['jy'] * df['x'])/np.sqrt( df['y']**2 + df['x']**2 )
    
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1] + ' (hhmmss)'

    return df, title

def rounded_cumsum_Bz():
    """Compute cumulative sum of dBx, dBy, and dBz to determine |B|.  
    Round the dB values to 6 significant digits to be comparable to the Paraview
    data.
    
    Inputs:
        None = data read from BATSRUS file is at origin + base + .out 
            (see above global constants)
    Outputs:
        None - logs |B| to output
     """
    
    # Read BATSRUS file
    df1, title = convert_BATSRUS_to_dataframe(base, origin)
    
    # Unfortunately, we can't use the dataframe cumsum because we have to 
    # round all of the entries.  So we do it the old fashion way, a loop
    n = len(df1)
    dBxRndSum = 0
    dByRndSum = 0
    dBzRndSum = 0
    
    for i in range(n):
        dBxRndSum = dBxRndSum + round_to_6(df1['dBx'][i])
        dByRndSum = dByRndSum + round_to_6(df1['dBy'][i])
        dBzRndSum = dBzRndSum + round_to_6(df1['dBz'][i])
        if( i%10000 == 0 ): logging.info(f'On cell: {i}')
    
    # Determine |B|
    BMag = np.sqrt( dBxRndSum**2 + dByRndSum**2 + dBzRndSum**2 )
    
    logging.info(f'Rounded BATSRUS |B| = {BMag}')
    return
    
def process_data_compare( reltol = 0.0001  ):
    """Compare data from BATSRUS file to Paraview file.  Read data in BATSRUS file 
    and Paraview file to create dataframes.  Compare quantities at randomly
    selected x,y,z points.  Since Paraview file is smaller than BATSRUS file, we
    randomly select point from Paraview and find corresponding point in BATSRUS.
        
    Inputs:
        n = number of points to compare between the BATSRUS and the Paraview files.
        reltol = relative tolerance used in np.isclose calls below.
        
        Note: 
        BATSRUS file is at origin + base + .out (see above global constants)
        Paraview file is at target + paraview (see above global constants)
    Outputs:
        None - other than plots
    """
    df1, title = convert_BATSRUS_to_dataframe(base, origin)
    
    logging.info('Parsing Paraview file...')

    # Verify Paraview file exists
    assert exists(target + paraview)

    df2 = pd.read_csv( target + paraview )
    n = len(df2)
    
    # Create storage for results ..1 variables are for the BATSRUS data and
    # ..2 variables are for Paraview data

    x1 = np.zeros(n, dtype=np.float32)
    y1 = np.zeros(n, dtype=np.float32)
    z1 = np.zeros(n, dtype=np.float32)
    
    bx1 = np.zeros(n, dtype=np.float32)
    by1 = np.zeros(n, dtype=np.float32)
    bz1 = np.zeros(n, dtype=np.float32)
    
    jx1 = np.zeros(n, dtype=np.float32)
    jy1 = np.zeros(n, dtype=np.float32)
    jz1 = np.zeros(n, dtype=np.float32)
    
    jphi1sc = np.zeros(n, dtype=np.float32)
    jphi1xy = np.zeros(n, dtype=np.float32)
    
    ux1 = np.zeros(n, dtype=np.float32)
    uy1 = np.zeros(n, dtype=np.float32)
    uz1 = np.zeros(n, dtype=np.float32)

    dbx1 = np.zeros(n, dtype=np.float32)
    dby1 = np.zeros(n, dtype=np.float32)
    dbz1 = np.zeros(n, dtype=np.float32)

    dbx1p = np.zeros(n, dtype=np.float32)
    dby1p = np.zeros(n, dtype=np.float32)
    dbz1p = np.zeros(n, dtype=np.float32)

    x2 = np.zeros(n, dtype=np.float32)
    y2 = np.zeros(n, dtype=np.float32)
    z2 = np.zeros(n, dtype=np.float32)
    
    bx2 = np.zeros(n, dtype=np.float32)
    by2 = np.zeros(n, dtype=np.float32)
    bz2 = np.zeros(n, dtype=np.float32)

    jx2 = np.zeros(n, dtype=np.float32)
    jy2 = np.zeros(n, dtype=np.float32)
    jz2 = np.zeros(n, dtype=np.float32)
    
    jphi2 = np.zeros(n, dtype=np.float32)
    
    ux2 = np.zeros(n, dtype=np.float32)
    uy2 = np.zeros(n, dtype=np.float32)
    uz2 = np.zeros(n, dtype=np.float32)

    dbx2 = np.zeros(n, dtype=np.float32)
    dby2 = np.zeros(n, dtype=np.float32)
    dbz2 = np.zeros(n, dtype=np.float32)

    # Pick n samples from the Paraview file and find the cooresponding entry
    # in the BATSRUS file.  We'll store matching records and plot the results below.
    for i in range(n):
        # Since the Paraview file is smaller, we select an entry from it
        # and find the corresponding point in the BATSRUS file
        if( i%100 == 0 ): logging.info(f'Examining point {i}...')
        df2_sample = df2.iloc[i]
        
        # extract the x,y,z position
        x2[i] = df2_sample['CellCenter_0']#.iloc[-1]
        y2[i] = df2_sample['CellCenter_1']#.iloc[-1]
        z2[i] = df2_sample['CellCenter_2']#.iloc[-1]
        
        # Look for cooresponding entry in BATSRUS data, use np.isclose
        # to find the x, then the y, and finally the z coordinate
        df1a = df1[ np.isclose( df1['x'], x2[i], rtol = reltol ) ]
        # logging.info(f'Returned x cut...{len(df1a)}')

        if( len(df1a) < 1):
            logging.info(f'Returned bad df1a...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
    
        df1b = df1a[ np.isclose( df1a['y'], y2[i], rtol = reltol ) ]
        # logging.info(f'Returned y cut...{len(df1b)}')

        if( len(df1b) < 1):
            logging.info(f'Returned bad df1b...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
    
        df1c = df1b[ np.isclose( df1b['z'], z2[i], rtol = reltol ) ]
        # logging.info(f'Returned z cut...{len(df1c)}')

        if( len(df1c) != 1):
            logging.info(f'Returned bad df1c...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
        
        # If we got thru the x,y,z search, we record the data and go to the
        # next i. 
        
        jx2[i] = df2_sample['j_0']#.iloc[-1]
        jy2[i] = df2_sample['j_1']#.iloc[-1]
        jz2[i] = df2_sample['j_2']#.iloc[-1]
        
        jphi2[i] = df2_sample['j_phi']#.iloc[-1]
        
        ux2[i] = df2_sample['u_0']#.iloc[-1]
        uy2[i] = df2_sample['u_1']#.iloc[-1]
        uz2[i] = df2_sample['u_2']#.iloc[-1]

        bx2[i] = df2_sample['b_0']#.iloc[-1]
        by2[i] = df2_sample['b_1']#.iloc[-1]
        bz2[i] = df2_sample['b_2']#.iloc[-1]

        # Paraview has incorrect constant, 673.1, and should be 637.1
        dbx2[i] = round_to_6(637.1/673.1*df2_sample['dB_0'])#.iloc[-1]
        dby2[i] = round_to_6(637.1/673.1*df2_sample['dB_1'])#.iloc[-1]
        dbz2[i] = round_to_6(637.1/673.1*df2_sample['dB_2'])#.iloc[-1]

        # The Paraview data has only 6 signficant digits, hence we need to 
        # round the BATSRUS data
        jx1[i] = round_to_6(df1c['jx'].iloc[-1])
        jy1[i] = round_to_6(df1c['jy'].iloc[-1])
        jz1[i] = round_to_6(df1c['jz'].iloc[-1])
                
        jphi1sc[i] = round_to_6(df1c['jphi-sc'].iloc[-1])
        jphi1xy[i] = round_to_6(df1c['jphi-xy'].iloc[-1])
               
        ux1[i] = round_to_6(df1c['ux'].iloc[-1])
        uy1[i] = round_to_6(df1c['uy'].iloc[-1])
        uz1[i] = round_to_6(df1c['uz'].iloc[-1])

        bx1[i] = round_to_6(df1c['bx'].iloc[-1])
        by1[i] = round_to_6(df1c['by'].iloc[-1])
        bz1[i] = round_to_6(df1c['bz'].iloc[-1])

        x1[i] = round_to_6(df1c['x'].iloc[-1])
        y1[i] = round_to_6(df1c['y'].iloc[-1])
        z1[i] = round_to_6(df1c['z'].iloc[-1])

        # Similarly, redo calculation of dBs using rounded quantities.
        # r = round_to_6(np.sqrt(x1[i]**2 + y1[i]**2 + z1[i]**2))
        # measure = round_to_6(df1c['measure'].iloc[-1])
        # factor = round_to_6(637.1*measure/r**3)
        # dbx1[i] = round_to_6(factor*( jy1[i]*(Z-z1[i]) - jz1[i]*(Y-y1[i]) ))
        # dby1[i] = round_to_6(factor*( jz1[i]*(X-x1[i]) - jx1[i]*(Z-z1[i]) ))
        # dbz1[i] = round_to_6(factor*( jx1[i]*(Y-y1[i]) - jy1[i]*(X-x1[i]) ))
              
        dbx1[i] = round_to_6(df1c['dBx'].iloc[-1])
        dby1[i] = round_to_6(df1c['dBy'].iloc[-1])
        dbz1[i] = round_to_6(df1c['dBz'].iloc[-1])

    # Plot results to compare them.  If the results are equivalent, we should
    # only see 45 degree diagonal lines.
    
    plt.subplot(1,3,1).scatter(x=x1, y=x2, s=5)
    plt.xlabel(r'$x$ (BATSRUS)')
    plt.ylabel(r'$x$ (Paraview)')
 
    plt.subplot(1,3,2).scatter(x=y1, y=y2, s=5)
    plt.xlabel(r'$y$ (BATSRUS)')
    plt.ylabel(r'$y$ (Paraview)')
 
    plt.subplot(1,3,3).scatter(x=z1, y=z2, s=5)
    plt.xlabel(r'$z$ (BATSRUS)')
    plt.ylabel(r'$z$ (Paraview)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=x1, y=diff_over_avg(x2,x1), s=5)
    plt.xlabel(r'$x$ (BATSRUS)')
    plt.ylabel(r'$2\frac{x(P)-x(B)}{x(P)+x(B)}$')
 
    plt.subplot(1,3,2).scatter(x=y1, y=diff_over_avg(y2,y1), s=5)
    plt.xlabel(r'$y$ (BATSRUS)')
    plt.ylabel(r'$2\frac{y(P)-y(B)}{y(P)+y(B)}$')
 
    plt.subplot(1,3,3).scatter(x=z1, y=diff_over_avg(z2,z1), s=5)
    plt.xlabel(r'$z$ (BATSRUS)')
    plt.ylabel(r'$2\frac{z(P)-z(B)}{z(P)+z(B)}$')

    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jx1, y=jx2, s=5)
    plt.xlabel(r'$j_x$ (BATSRUS)')
    plt.ylabel(r'$j_x$ (Paraview)')
    
    plt.subplot(1,3,2).scatter(x=jy1, y=jy2, s=5)
    plt.xlabel(r'$j_y$ (BATSRUS)')
    plt.ylabel(r'$j_y$ (Paraview)')
    
    plt.subplot(1,3,3).scatter(x=jz1, y=jz2, s=5)
    plt.xlabel(r'$j_z$ (BATSRUS)')
    plt.ylabel(r'$j_z$ (Paraview)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=jx1, y=diff_over_avg(jx2,jx1), s=5)
    plt.xlabel(r'$j_x$ (BATSRUS)')
    plt.ylabel(r'$2\frac{j_x(P)-j_x(B)}{j_x(P)+j_x(B)}$')
 
    plt.subplot(1,3,2).scatter(x=jy1, y=diff_over_avg(jy2,jy1), s=5)
    plt.xlabel(r'$j_y$ (BATSRUS)')
    plt.ylabel(r'$2\frac{j_y(P)-j_y(B)}{j_y(P)+j_y(B)}$')
 
    plt.subplot(1,3,3).scatter(x=jz1, y=diff_over_avg(jz2,jz1), s=5)
    plt.xlabel(r'$j_z$ (BATSRUS)')
    plt.ylabel(r'$2\frac{j_z(P)-j_z(B)}{j_z(P)+j_z(B)}$')

    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jphi1sc, y=jphi2, s=5)
    plt.xlabel(r'$j_\phi$ (sin-cos) (BATSRUS)')
    plt.ylabel(r'$j_\phi$ (Paraview)')
    plt.title('Using sin and cos')
    
    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jphi1sc, y=diff_over_avg(jphi2,jphi1sc), s=5)
    plt.xlabel(r'$j_\phi$ (sin-cos) (BATSRUS)')
    plt.ylabel(r'$2 \frac{j_\phi (P) - j_\phi (B)}{j_\phi (P) + j_\phi (B)}$')
    plt.title('Using sin and cos')

    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jphi1xy, y=jphi2, s=5)
    plt.xlabel(r'$j_\phi$ (x-y) (BATSRUS)')
    plt.ylabel(r'$j_\phi$ (Paraview)')
    plt.title('Using x and y')
    
    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jphi1xy, y=diff_over_avg(jphi2, jphi1xy), s=5)
    plt.xlabel(r'$j_\phi$ (x-y) (BATSRUS)')
    plt.ylabel(r'$2 \frac{j_\phi (P) - j_\phi (B)}{j_\phi (P) + j_\phi (B)}$')
    plt.title('Using x and y')
     
    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=jphi1xy, y=diff_over_avg(jphi1sc, jphi1xy), s=5)
    plt.xlabel(r'$j_\phi$ (x-y) (B)')
    plt.ylabel(r'$2 \frac{j_\phi (sin-cos)(B) - j_\phi (x-y)(B)}{j_\phi (sin-cos)(B) + j_\phi (x-y)(B)}$')
    plt.title('sin-cos vs x-y')
     
    plt.figure()

    plt.subplot(1,3,1).scatter(x=ux1, y=ux2, s=5)
    plt.xlabel(r'$u_x$ (BATSRUS)')
    plt.ylabel(r'$u_x$ (Paraview)')
    
    plt.subplot(1,3,2).scatter(x=uy1, y=uy2, s=5)
    plt.xlabel(r'$u_y$ (BATSRUS)')
    plt.ylabel(r'$u_y$ (Paraview)')
    
    plt.subplot(1,3,3).scatter(x=uz1, y=uz2, s=5)
    plt.xlabel(r'$u_z$ (BATSRUS)')
    plt.ylabel(r'$u_z$ (Paraview)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=ux1, y=diff_over_avg(ux2, ux1), s=5)
    plt.xlabel(r'$u_x$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{u_x (P) - u_x (B)}{u_x (P) + u_x (B)}$')
   
    plt.subplot(1,3,2).scatter(x=uy1, y=diff_over_avg(uz2, uz1), s=5)
    plt.xlabel(r'$u_y$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{u_y (P) - u_y (B)}{u_y (P) + u_y (B)}$')
    
    plt.subplot(1,3,3).scatter(x=uz1, y=diff_over_avg(uz2, uz1), s=5)
    plt.xlabel(r'$u_z$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{u_z (P) - u_z (B)}{u_z (P) + u_z (B)}$')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=bx1, y=bx2, s=5)
    plt.xlabel(r'$B_x$ (BATSRUS)')
    plt.ylabel(r'$B_x$ (Paraview)')
    
    plt.subplot(1,3,2).scatter(x=by1, y=by2, s=5)
    plt.xlabel(r'$B_y$ (BATSRUS)')
    plt.ylabel(r'$B_y$ (Paraview)')
    
    plt.subplot(1,3,3).scatter(x=bz1, y=bz2, s=5)
    plt.xlabel(r'$B_z$ (BATSRUS)')
    plt.ylabel(r'$B_z$ (Paraview)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=bx1, y=diff_over_avg(bx2, bx1), s=5)
    plt.xlabel(r'$B_x$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{B_x (P) - B_x (B)}{B_x (P) + B_x (B)}$')
   
    plt.subplot(1,3,2).scatter(x=by1, y=diff_over_avg(bz2, bz1), s=5)
    plt.xlabel(r'$B_y$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{B_y (P) - B_y (B)}{B_y (P) + B_y (B)}$')
    
    plt.subplot(1,3,3).scatter(x=bz1, y=diff_over_avg(bz2, bz1), s=5)
    plt.xlabel(r'$B_z$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{B_z (P) - B_z (B)}{B_z (P) + B_z (B)}$')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=dbx1, y=dbx2, s=5)
    plt.xlabel(r'$\delta B_x$ (BATSRUS)')
    plt.ylabel(r'$\delta B_x$ (Paraview)')
    
    plt.subplot(1,3,2).scatter(x=dby1, y=dby2, s=5)
    plt.xlabel(r'$\delta B_y$ (BATSRUS)')
    plt.ylabel(r'$\delta B_y$ (Paraview)')
    
    plt.subplot(1,3,3).scatter(x=dbz1, y=dbz2, s=5)
    plt.xlabel(r'$\delta B_z$ (BATSRUS)')
    plt.ylabel(r'$\delta B_z$ (Paraview)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=dbx1, y=diff_over_avg(dbx2, dbx1), s=5)
    plt.xlabel(r'$\delta B_x$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{\delta B_x (P) - \delta B_x (B)}{\delta B_x (P) + \delta B_x (B)}$')
   
    plt.subplot(1,3,2).scatter(x=dby1, y=diff_over_avg(dbz2, dbz1), s=5)
    plt.xlabel(r'$\delta B_y$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{\delta B_y (P) - \delta B_y (B)}{\delta B_y (P) + \delta B_y (B)}$')
    
    plt.subplot(1,3,3).scatter(x=dbz1, y=diff_over_avg(dbz2, dbz1), s=5)
    plt.xlabel(r'$\delta B_z$ (BATSRUS)')
    plt.ylabel(r'$2 \frac{\delta B_z (P) - \delta B_z (B)}{\delta B_z (P) + \delta B_z (B)}$')
    
if __name__ == "__main__":
      process_data_compare()
     # rounded_cumsum_Bz()
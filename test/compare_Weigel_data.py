#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:39:37 2022

@author: Dean Thomas
"""

##############################################################################
##############################################################################
# This code compares my Biot-Savart code for calculating delta B to Weigel's
# Paraview code that does the same thing.  Differences should be small, 10^-5 or
# 10^-6 based on rounding of the data from Paraview to six significant digits.
#
# See plots created by process_data_compare to do comparison.  Differences
# from diff_over_avg should be small (10^-5 to 10^-6)
##############################################################################
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import exists
import logging

# origin and target define where input data and output plots are stored
origin = '/Volumes/Physics HD v2/divB_simple1/GM/'
target = '/Volumes/Physics HD v2/divB_simple1/data_comparison/'

# names of BATSRUS and Paraview file
base = '3d__mhd_4_e20100320-000000-000'
paraview = 'weigel_data.csv'

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
rCurrents = 3.

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

from deltaB import convert_BATSRUS_to_dataframe, create_deltaB_rCurrents_dataframe

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
    
    # Read BATSRUS fileDocuments
    filename = origin + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents)
    df1 = create_deltaB_rCurrents_dataframe(df1, [X,Y,Z])
    
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
    filename = origin + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents)
    df1 = create_deltaB_rCurrents_dataframe(df1, [X,Y,Z])

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
    
    # jphi1sc = np.zeros(n, dtype=np.float32)
    # jphi1xy = np.zeros(n, dtype=np.float32)
    
    ux1 = np.zeros(n, dtype=np.float32)
    uy1 = np.zeros(n, dtype=np.float32)
    uz1 = np.zeros(n, dtype=np.float32)

    dbx1 = np.zeros(n, dtype=np.float32)
    dby1 = np.zeros(n, dtype=np.float32)
    dbz1 = np.zeros(n, dtype=np.float32)

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
        
        ##################################################################
        #!!!! Paraview has incorrect constant, 673.1, and should be 637.1
        ##################################################################
        dbx2[i] = round_to_6(637.1/673.1*df2_sample['dB_0'])#.iloc[-1]
        dby2[i] = round_to_6(637.1/673.1*df2_sample['dB_1'])#.iloc[-1]
        dbz2[i] = round_to_6(637.1/673.1*df2_sample['dB_2'])#.iloc[-1]

        # The Paraview data has only 6 signficant digits, hence we need to 
        # round the BATSRUS data
        jx1[i] = round_to_6(df1c['jx'].iloc[-1])
        jy1[i] = round_to_6(df1c['jy'].iloc[-1])
        jz1[i] = round_to_6(df1c['jz'].iloc[-1])
                
        # jphi1sc[i] = round_to_6(df1c['jphi-sc'].iloc[-1])
        # jphi1xy[i] = round_to_6(df1c['jphi-xy'].iloc[-1])
               
        ux1[i] = round_to_6(df1c['ux'].iloc[-1])
        uy1[i] = round_to_6(df1c['uy'].iloc[-1])
        uz1[i] = round_to_6(df1c['uz'].iloc[-1])

        bx1[i] = round_to_6(df1c['bx'].iloc[-1])
        by1[i] = round_to_6(df1c['by'].iloc[-1])
        bz1[i] = round_to_6(df1c['bz'].iloc[-1])

        x1[i] = round_to_6(df1c['x'].iloc[-1])
        y1[i] = round_to_6(df1c['y'].iloc[-1])
        z1[i] = round_to_6(df1c['z'].iloc[-1])

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
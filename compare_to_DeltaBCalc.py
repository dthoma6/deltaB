#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:13:37 2023

@author: Dean Thomas
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deltaB.BATSRUS_dataframe import calc_gap_dB, convert_BATSRUS_to_dataframe, \
    create_cumulative_sum_dataframe
from deltaB.util import ned, date_time, date_timeISO, get_files

from spacepy import coordinates as coord
from spacepy.time import Ticktock

import logging

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

ROUNDED = False
SHIFTED = False
RCURR = False

Re = 6378.137
if RCURR:
    RCURRENTS = 2.
else:
    RCURRENTS = 1.8
rIonosphere = 1.01725

# IFILE='position_18.10_72.00.txt'
# BFILE='B_Magnetosphere_18.10_72.pkl'
# XPOS=[(Re + 1.)/Re, 18.1034, 72.0000]

# IFILE='position_18.10_252.00.txt'
# BFILE='B_Magnetosphere__18.10_252.00.pkl'
# if SHIFTED:
#     DIFF = 0.5
#     XPOS=[(Re + 1.)/Re, 18.1034+DIFF, 252.0000+DIFF]
# else:
#     XPOS=[(Re + 1.)/Re, 18.1034, 252.0000]

IFILE='position_18.91_72.81.txt'
BFILE='B_Magnetosphere_18.91_72.81.pkl'
XPOS=[(Re + 1.)/Re, 18.9070, 72.8150]

SKIPFILES=True

# origin and target define where input data and output plots are stored
ORIGIN = '/Volumes/Physics HD v2/runs/DIPTSUR2/GM/IO2/'
TARGET = '/Volumes/Physics HD v2/runs/DIPTSUR2/data_comparison/'

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

def rounded_cumsum_B(X, base, dirpath, rCurrents):
    """Compute cumulative sum of dBx, dBy, and dBz to determine |B|.  
    Round the dB values to 6 significant digits to be comparable to the Paraview
    data.
    
    Inputs:
        X = point where magnetic field estimated using Biot-Savart
        
        base, dirpath = data read from BATSRUS file is at dirpath + base + .out 
        
        rCurrents = rCurrents from SWMF
        
    Outputs:
        dBxRndSum, dByRndSum, dBzRndSum = x,y,z components of B field using rounded values
     """
    
    # Read BATSRUS file
    df1, title = convert_BATSRUS_to_dataframe(X, base, dirpath=dirpath, rCurrents=rCurrents)
    
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
        # if( i%10000 == 0 ): logging.info(f'On cell: {i}')
    
    return dBxRndSum, dByRndSum, dBzRndSum

def compare_magnetosphere():
    
    Xlatlong = XPOS

    files = get_files(orgdir=ORIGIN, reduce=SKIPFILES, base='3d__var_2_e*')

    n = len(files)
    
    y = np.zeros(n)
    m = np.zeros(n)
    d = np.zeros(n)
    hh = np.zeros(n)
    mm = np.zeros(n)
    ss = np.zeros(n)
    bn = np.zeros(n)
    be = np.zeros(n)
    bd = np.zeros(n)
    
    for i in range(n):
        
        timeiso = date_timeISO(files[i])
        y[i], m[i], d[i], hh[i], mm[i], ss[i] = date_time(files[i])
        
        Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph')
        Xgeo.ticks = Ticktock([timeiso], 'ISO')
        Xgsm = Xgeo.convert('GSM', 'car')
        X = Xgsm.data[0]
     
        n_geo, e_geo, d_geo = ned(timeiso, X, 'GSM')
        
        if ROUNDED:
            Bx, By, Bz = rounded_cumsum_B(X, files[i], ORIGIN, RCURRENTS)
            
            bn[i] = Bx*n_geo[0] + By*n_geo[1] + Bz*n_geo[2]
            be[i] = Bx*e_geo[0] + By*e_geo[1] + Bz*e_geo[2]
            bd[i] = Bx*d_geo[0] + By*d_geo[1] + Bz*d_geo[2]
        else:
            df, title = convert_BATSRUS_to_dataframe(X, files[i], dirpath=ORIGIN, rCurrents=RCURRENTS)   
            df_r = create_cumulative_sum_dataframe(df)
        
            bn[i] = df_r['dBxSum'].iloc[-1]*n_geo[0] + \
                df_r['dBySum'].iloc[-1]*n_geo[1] + \
                df_r['dBzSum'].iloc[-1]*n_geo[2]
        
            be[i] = df_r['dBxSum'].iloc[-1]*e_geo[0] + \
                df_r['dBySum'].iloc[-1]*e_geo[1] + \
                df_r['dBzSum'].iloc[-1]*e_geo[2]
        
            bd[i] = df_r['dBxSum'].iloc[-1]*d_geo[0] + \
                df_r['dBySum'].iloc[-1]*d_geo[1] + \
                df_r['dBzSum'].iloc[-1]*d_geo[2]
            
    df = pd.DataFrame()
    
    df['year'] = y
    df['month'] = m
    df['day'] = d
    df['hour'] = hh
    df['minute'] = mm
    df['second'] = ss
    df['B_north_mag'] = bn
    df['B_east_mag'] = be
    df['B_down_mag'] = bd
    
    if RCURR:
        df.to_pickle(TARGET + 'rCurrent' + BFILE)
    elif SHIFTED:
        df.to_pickle(TARGET + 'Shifted' + BFILE)
    elif ROUNDED:
        df.to_pickle(TARGET + 'Rounded' + BFILE)
    else:
        df.to_pickle(TARGET + BFILE)
    
def read_ccmc_printout(filename):
    '''
    Reads files with text like ::
    # Data printout from CCMC-simulation: version 1D-1.3
    # Data type:  CalcDeltaB  PP
    # Run name:   Gary_Quaresima_20210809_PP_1
    # Missing data:  -1.09951e+12
    # year month day hour minute second lon lat alt B_north B_east B_down B_north_mag B_east_mag B_down_mag B_north_fac B_east_fac B_down_fac B_north_iono,SigP B_east_iono,SigP B_down_iono,SigP B_north_iono,SigH B_east_iono,SigH B_down_iono,SigH 
    # year month day hr min s deg deg km nT nT nT nT nT nT nT nT nT nT nT nT nT nT nT 
      2.01900E+03  9.00000E+00  2.00000E+00  4.00000E+00  1.00000E+01  0.00000E+00  7.28150E+01  1.89070E+01  1.00000E+00  6.62460E+00 -2.46310E+00  2.40840E+00  6.11990E+00 -3.47180E+00  2.15050E+00 -9.67200E-01  2.64940E+00  1.41000E-01  4.60600E-01  1.05700E-01 -2.07000E-02 -1.93000E-01  3.18400E-01 -8.70000E-01
      ...
    '''
    with open(filename) as f:
        head = [f.readline() for _ in range(5)]
        arr = np.loadtxt(f)
    headers = tuple(head[-1].split(' ')[1:-1])
    return headers, arr
    
def compare_to_CalcDeltaB():
    headers, data = read_ccmc_printout( TARGET +  IFILE)
    
    df1 = pd.DataFrame()
    
    df1['hour'] = data[:,3]
    df1['hour'] = df1['hour'].astype(float)
    df1['minute'] = data[:,4]
    df1['minute'] = df1['minute'].astype(float)
    df1['second'] = data[:,5]
    df1['second'] = df1['second'].astype(float)
    
    df1['time'] = df1['hour'] + df1['minute']/60. + df1['second']/3600.
    
    df1['B_north CalcDeltaB'] = data[:,12]
    df1['B_north CalcDeltaB'] = df1['B_north CalcDeltaB'].astype(float)
    df1['B_east CalcDeltaB'] = data[:,13]
    df1['B_east CalcDeltaB'] = df1['B_east CalcDeltaB'].astype(float)
    df1['B_down CalcDeltaB'] = data[:,14]
    df1['B_down CalcDeltaB'] = df1['B_down CalcDeltaB'].astype(float)
    df1['B_mag CalcDeltaB'] = np.sqrt(df1['B_north CalcDeltaB']**2 + 
                                      df1['B_east CalcDeltaB']**2 + 
                                      df1['B_down CalcDeltaB']**2)
    
    if RCURR:
        df2 = pd.read_pickle( TARGET + 'rCurrent' + BFILE)
        title = "rCurr " + BFILE
    elif SHIFTED:
        df2 = pd.read_pickle( TARGET + 'Shifted' + BFILE)
        title = "Shift " + BFILE
    elif ROUNDED:
        df2 = pd.read_pickle( TARGET + 'Rounded' + BFILE)
        title = "Rnd " + BFILE
    else:   
        df2 = pd.read_pickle( TARGET + BFILE)
        title = BFILE
    df2['time'] = df2['hour'] + df2['minute']/60. + df2['second']/3600.
    df2['B_mag'] = np.sqrt(df2['B_north_mag']**2 + 
                                      df2['B_east_mag']**2 + 
                                      df2['B_down_mag']**2)
      
    # plot lines
    plt.plot(df1['time'], df1['B_north CalcDeltaB'], label = "Bn CalcDeltaB")
    plt.plot(df2['time'], df2['B_north_mag'], label = "Bn Mine")
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(df1['time'], df1['B_east CalcDeltaB'], label = "Be CalcDeltaB")
    plt.plot(df2['time'], df2['B_east_mag'], label = "Be Mine")
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(df1['time'], df1['B_down CalcDeltaB'], label = "Bd CalcDeltaB")
    plt.plot(df2['time'], df2['B_down_mag'], label = "Bd Mine")
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(df1['time'], df1['B_mag CalcDeltaB'], label = "|B| CalcDeltaB")
    plt.plot(df2['time'], df2['B_mag'], label = "|B| Mine")
    plt.title(title)
    plt.legend()
    plt.show()

    

if __name__ == "__main__":
    # main(sys.argv[1:])
    compare_magnetosphere()
    compare_to_CalcDeltaB()

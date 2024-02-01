#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 15:54:50 2024

@author: Dean Thomas
"""

####################################################################
####################################################################
#
# Comparison of CCMC, magnetopost, and deltaB methods of calculating total
# B at a specific point on earth.  
#
####################################################################
####################################################################


import magnetopost as mp
import os.path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#############################################################
# Code borrowed from magnetopost to read CCMC files
#############################################################

ned = ('north','east','down')

def read_ccmc_printout(filename):
    '''
    Reads files with text like ::
    # Data printout from CCMC-simulation: version 1D-1.3
    # Data type:  CCMC  PP
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
    # headers = tuple(head[-1].split(' ')[1:-1])
    headers = tuple(head[-1].split()[1:])
    return headers, arr

def extract_from_swmf_ccmc_printout_file(filename):

    mp.logger.info("Reading point dB information from " + filename)
    headers, arr = read_ccmc_printout(filename)
    assert( headers[10:] ==('sumBn', 'sumBe', 'sumBd',
                             'dBn', 'dBe', 'dBd',
                            'facdBn', 'facdBe', 'facdBd',
                            'JhdBn', 'JhdBe', 'JhdBd',
                            'JpBn', 'JpBe', 'JpBd') )

    mp.logger.info("Read point dB information from " + filename)

    times = np.array(arr[:,0:6], dtype=int)
    dtimes = [datetime(*time) for time in times]

    dBMhd = pd.DataFrame(data=arr[:,13:16], columns=ned, index=dtimes)
    dBFac = pd.DataFrame(data=arr[:,16:19], columns=ned, index=dtimes)
    dBHal = pd.DataFrame(data=arr[:,19:22], columns=ned, index=dtimes)
    dBPed = pd.DataFrame(data=arr[:,22:25], columns=ned, index=dtimes)

    return dBMhd, dBFac, dBHal, dBPed

def extract_from_magnetopost_files(info, surface_location):

    msph_times = info['files']['magnetosphere'].keys()
    iono_times = info['files']['ionosphere'].keys()

    msph_dtimes = [datetime(*time) for time in msph_times]
    iono_dtimes = [datetime(*time) for time in iono_times]

    def get(ftag, dtimes):
        df = pd.DataFrame()
        infile = f'../data/{ftag}-{surface_location}.npy'
        mp.logger.info(f"Reading {infile}")
        dB = np.load(infile)

        df['north'] = pd.Series(data=dB[:,0], index=dtimes)
        df['east']  = pd.Series(data=dB[:,1], index=dtimes)
        df['down']  = pd.Series(data=dB[:,2], index=dtimes)
        return df

    bs_msph     = get('bs_msph', msph_dtimes)
    bs_fac      = get('bs_fac', msph_dtimes)

    bs_hall     = get('bs_hall', iono_dtimes)
    bs_pedersen = get('bs_pedersen', iono_dtimes)

    return bs_msph, bs_fac, bs_hall, bs_pedersen


#############################################################
# Workhorse for plotting
#############################################################

def ms_plots(info, point):

    # Read the CCMC data
    cdbfilepath = info['deltaB_files'][point]
    dBMhd, dBFac, dBHal, dBPed = extract_from_swmf_ccmc_printout_file( cdbfilepath )

    # Read the magnetopost data
    bs_msph, bs_fac, bs_hall, bs_pedersen = extract_from_magnetopost_files(info, point)
    
    # Read deltaB data
    pklname = 'dB_bs_msph-' + point + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['mag']      = np.sqrt(df['Bn']**2         + df['Be']**2        + df['Bd']**2)
    dBMhd['mag']   = np.sqrt(dBMhd['north']**2   + dBMhd['east']**2   + dBMhd['down']**2)
    bs_msph['mag'] = np.sqrt(bs_msph['north']**2 + bs_msph['east']**2 + bs_msph['down']**2)

    # Combine the three dataframes into one.  
    # NOTE, all the dataframes use the file timestamps to index the entries, 
    dBMhd.columns =['Bn_cdb','Be_cdb','Bd_cdb','mag_cdb']        # CCMC
    bs_msph.columns =['Bn_mp','Be_mp','Bd_mp','mag_mp']          # magnetopost
    df_all = dBMhd.merge( df, left_index=True, right_index=True) # deltaB
    df_all = df_all.merge( bs_msph, left_index=True, right_index=True) 
    
    # Calculate differences between the B fields from the three methods.
    df_all['cdb-db north'] = df_all['Bn_cdb'] - df_all['Bn']
    df_all['cdb-db east'] = df_all['Be_cdb'] - df_all['Be']
    df_all['cdb-db down'] = df_all['Bd_cdb'] - df_all['Bd'] 
    df_all['cdb-db mag'] = df_all['mag_cdb'] - df_all['mag']
 
    df_all['mp-db north'] = df_all['Bn_mp'] - df_all['Bn']
    df_all['mp-db east'] = df_all['Be_mp'] - df_all['Be']
    df_all['mp-db down'] = df_all['Bd_mp'] - df_all['Bd'] 
    df_all['mp-db mag'] = df_all['mag_mp'] - df_all['mag']
 
    # Plot results
    ax = df_all.plot(y='Bn_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bn_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bn', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Magnetosphere $B_N$' )
     
    ax = df_all.plot(y='Be_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Be_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Be', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Magnetosphere $B_E$' )
    
    ax = df_all.plot(y='Bd_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bd_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bd', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Magnetosphere $B_D$' )
 
    ax = df_all.plot(y='mag_cdb', label='CCMC')
    df_all.plot( ax=ax, y='mag_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='mag', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Magnetosphere $|B|$' )

    # ax = df_all.plot(y='cdb-db north', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db north', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Magnetosphere $\Delta B_N$' )

    # ax = df_all.plot(y='cdb-db east', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db east', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Magnetosphere $\Delta B_E$' )

    # ax = df_all.plot(y='cdb-db down', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db down', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Magnetosphere $\Delta B_D$' )

    # ax = df_all.plot(y='cdb-db mag', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db mag', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Magnetosphere $\Delta |B|$' )

    print('Mean CCMC-deltaB Magnetosphere Bn: ', df_all['cdb-db north'].mean())
    print('Mean CCMC-deltaB Magnetosphere Be: ', df_all['cdb-db east'].mean())
    print('Mean CCMC-deltaB Magnetosphere Bd: ', df_all['cdb-db down'].mean())
    print('Mean CCMC-deltaB Magnetosphere |B|: ', df_all['cdb-db down'].mean())
    print('Std Dev CCMC-deltaB Magnetosphere Bn: ', df_all['cdb-db north'].std())
    print('Std Dev CCMC-deltaB Magnetosphere Be: ', df_all['cdb-db east'].std())
    print('Std Dev CCMC-deltaB Magnetosphere Bd: ', df_all['cdb-db down'].std())
    print('Std Dev CCMC-deltaB Magnetosphere |B|: ', df_all['cdb-db mag'].std())

    print('Mean Magnetopost-deltaB Magnetosphere Bn: ', df_all['mp-db north'].mean())
    print('Mean Magnetopost-deltaB Magnetosphere Be: ', df_all['mp-db east'].mean())
    print('Mean Magnetopost-deltaB Magnetosphere Bd: ', df_all['mp-db down'].mean())
    print('Mean Magnetopost-deltaB Magnetosphere |B|: ', df_all['mp-db down'].mean())
    print('Std Dev Magnetopost-deltaB Magnetosphere Bn: ', df_all['mp-db north'].std())
    print('Std Dev Magnetopost-deltaB Magnetosphere Be: ', df_all['mp-db east'].std())
    print('Std Dev Magnetopost-deltaB Magnetosphere Bd: ', df_all['mp-db down'].std())
    print('Std Dev Magnetopost-deltaB Magnetosphere |B|: ', df_all['mp-db mag'].std())
    return

def gap_plots(info, point):

    # Read the CCMC data
    cdbfilepath = info['deltaB_files'][point]
    dBMhd1, dBFac, dBHal, dBPed = extract_from_swmf_ccmc_printout_file( cdbfilepath )

    # Read the magnetopost data
    bs_msph1, bs_fac, bs_hall, bs_pedersen = extract_from_magnetopost_files(info, point)
    
    # Read deltaB data
    pklname = 'dB_bs_gap-' + point + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['mag']      = np.sqrt(df['Bn']**2         + df['Be']**2        + df['Bd']**2)
    dBFac['mag']   = np.sqrt(dBFac['north']**2   + dBFac['east']**2   + dBFac['down']**2)
    bs_fac['mag']  = np.sqrt(bs_fac['north']**2  + bs_fac['east']**2  + bs_fac['down']**2)

    # Combine the three dataframes into one.  
    # NOTE, all the dataframes use the file timestamps to index the entries, 
    dBFac.columns =['Bn_cdb','Be_cdb','Bd_cdb','mag_cdb']        # CCMC
    bs_fac.columns =['Bn_mp','Be_mp','Bd_mp','mag_mp']           # magnetopost
    df_all = dBFac.merge( df, left_index=True, right_index=True) # deltaB
    df_all = df_all.merge( bs_fac, left_index=True, right_index=True) 
    
    # Calculate differences between the B fields from the three methods.
    df_all['cdb-db north'] = df_all['Bn_cdb'] - df_all['Bn']
    df_all['cdb-db east'] = df_all['Be_cdb'] - df_all['Be']
    df_all['cdb-db down'] = df_all['Bd_cdb'] - df_all['Bd'] 
    df_all['cdb-db mag'] = df_all['mag_cdb'] - df_all['mag']
 
    df_all['mp-db north'] = df_all['Bn_mp'] - df_all['Bn']
    df_all['mp-db east'] = df_all['Be_mp'] - df_all['Be']
    df_all['mp-db down'] = df_all['Bd_mp'] - df_all['Bd'] 
    df_all['mp-db mag'] = df_all['mag_mp'] - df_all['mag']
 
    # Plot results
    ax = df_all.plot(y='Bn_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bn_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bn', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Gap $B_N$' )
     
    ax = df_all.plot(y='Be_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Be_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Be', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Gap $B_E$' )
    
    ax = df_all.plot(y='Bd_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bd_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bd', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Gap $B_D$' )
 
    ax = df_all.plot(y='mag_cdb', label='CCMC')
    df_all.plot( ax=ax, y='mag_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='mag', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Gap $|B|$' )

    # ax = df_all.plot(y='cdb-db north', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db north', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Gap $\Delta B_N$' )

    # ax = df_all.plot(y='cdb-db east', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db east', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Gap $\Delta B_E$' )

    # ax = df_all.plot(y='cdb-db down', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db down', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Gap $\Delta B_D$' )

    # ax = df_all.plot(y='cdb-db mag', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db mag', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Gap $\Delta |B|$' )

    print('Mean CCMC-deltaB Gap Bn: ', df_all['cdb-db north'].mean())
    print('Mean CCMC-deltaB Gap Be: ', df_all['cdb-db east'].mean())
    print('Mean CCMC-deltaB Gap Bd: ', df_all['cdb-db down'].mean())
    print('Mean CCMC-deltaB Gap |B|: ', df_all['cdb-db down'].mean())
    print('Std Dev CCMC-deltaB Gap Bn: ', df_all['cdb-db north'].std())
    print('Std Dev CCMC-deltaB Gap Be: ', df_all['cdb-db east'].std())
    print('Std Dev CCMC-deltaB Gap Bd: ', df_all['cdb-db down'].std())
    print('Std Dev CCMC-deltaB Gap |B|: ', df_all['cdb-db mag'].std())

    print('Mean Magnetopost-deltaB Gap Bn: ', df_all['mp-db north'].mean())
    print('Mean Magnetopost-deltaB Gap Be: ', df_all['mp-db east'].mean())
    print('Mean Magnetopost-deltaB Gap Bd: ', df_all['mp-db down'].mean())
    print('Mean Magnetopost-deltaB Gap |B|: ', df_all['mp-db down'].mean())
    print('Std Dev Magnetopost-deltaB Gap Bn: ', df_all['mp-db north'].std())
    print('Std Dev Magnetopost-deltaB Gap Be: ', df_all['mp-db east'].std())
    print('Std Dev Magnetopost-deltaB Gap Bd: ', df_all['mp-db down'].std())
    print('Std Dev Magnetopost-deltaB Gap |B|: ', df_all['mp-db mag'].std())
    return

def pedersen_plots(info, point):

    # Read the CCMC data
    cdbfilepath = info['deltaB_files'][point]
    dBMhd1, dBFac1, dBHal, dBPed = extract_from_swmf_ccmc_printout_file( cdbfilepath )

    # Read the magnetopost data
    bs_msph1, bs_fac1, bs_hall, bs_pedersen = extract_from_magnetopost_files(info, point)
    
    # Read deltaB data
    pklname = 'dB_bs_iono-' + point + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['mag']      = np.sqrt(df['Bnp']**2         + df['Bep']**2        + df['Bdp']**2)
    dBPed['mag']   = np.sqrt(dBPed['north']**2   + dBPed['east']**2   + dBPed['down']**2)
    bs_pedersen['mag']  = np.sqrt(bs_pedersen['north']**2  + bs_pedersen['east']**2  + bs_pedersen['down']**2)

    # Combine the three dataframes into one.  
    # NOTE, all the dataframes use the file timestamps to index the entries, 
    dBPed.columns =['Bn_cdb','Be_cdb','Bd_cdb','mag_cdb']        # CCMC
    bs_pedersen.columns =['Bn_mp','Be_mp','Bd_mp','mag_mp']      # magnetopost
    df_all = dBPed.merge( df, left_index=True, right_index=True) # deltaB
    df_all = df_all.merge( bs_pedersen, left_index=True, right_index=True) 
   
    # Calculate differences between the B fields from the three methods.
    df_all['cdb-db north'] = df_all['Bn_cdb'] - df_all['Bnp']
    df_all['cdb-db east'] = df_all['Be_cdb'] - df_all['Bep']
    df_all['cdb-db down'] = df_all['Bd_cdb'] - df_all['Bdp'] 
    df_all['cdb-db mag'] = df_all['mag_cdb'] - df_all['mag']
 
    df_all['mp-db north'] = df_all['Bn_mp'] - df_all['Bnp']
    df_all['mp-db east'] = df_all['Be_mp'] - df_all['Bep']
    df_all['mp-db down'] = df_all['Bd_mp'] - df_all['Bdp'] 
    df_all['mp-db mag'] = df_all['mag_mp'] - df_all['mag']

    # Plot results
    ax = df_all.plot(y='Bn_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bn_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bnp', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Pedersen $B_N$' )
     
    ax = df_all.plot(y='Be_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Be_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bep', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Pedersen $B_E$' )
    
    ax = df_all.plot(y='Bd_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bd_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bdp', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Pedersen $B_D$' )
 
    ax = df_all.plot(y='mag_cdb', label='CCMC')
    df_all.plot( ax=ax, y='mag_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='mag', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Pedersen $|B|$' )

    # ax = df_all.plot(y='cdb-db north', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db north', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Pedersen $\Delta B_N$' )

    # ax = df_all.plot(y='cdb-db east', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db east', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Pedersen $\Delta B_E$' )

    # ax = df_all.plot(y='cdb-db down', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db down', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Pedersen $\Delta B_D$' )

    # ax = df_all.plot(y='cdb-db mag', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db mag', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Pedersen $\Delta |B|$' )

    print('Mean CCMC-deltaB Pedersen Bn: ', df_all['cdb-db north'].mean())
    print('Mean CCMC-deltaB Pedersen Be: ', df_all['cdb-db east'].mean())
    print('Mean CCMC-deltaB Pedersen Bd: ', df_all['cdb-db down'].mean())
    print('Mean CCMC-deltaB Pedersen |B|: ', df_all['cdb-db down'].mean())
    print('Std Dev CCMC-deltaB Pedersen Bn: ', df_all['cdb-db north'].std())
    print('Std Dev CCMC-deltaB Pedersen Be: ', df_all['cdb-db east'].std())
    print('Std Dev CCMC-deltaB Pedersen Bd: ', df_all['cdb-db down'].std())
    print('Std Dev CCMC-deltaB Pedersen |B|: ', df_all['cdb-db mag'].std())

    print('Mean Magnetopost-deltaB Pedersen Bn: ', df_all['mp-db north'].mean())
    print('Mean Magnetopost-deltaB Pedersen Be: ', df_all['mp-db east'].mean())
    print('Mean Magnetopost-deltaB Pedersen Bd: ', df_all['mp-db down'].mean())
    print('Mean Magnetopost-deltaB Pedersen |B|: ', df_all['mp-db down'].mean())
    print('Std Dev Magnetopost-deltaB Pedersen Bn: ', df_all['mp-db north'].std())
    print('Std Dev Magnetopost-deltaB Pedersen Be: ', df_all['mp-db east'].std())
    print('Std Dev Magnetopost-deltaB Pedersen Bd: ', df_all['mp-db down'].std())
    print('Std Dev Magnetopost-deltaB Pedersen |B|: ', df_all['mp-db mag'].std())
    return

def hall_plots(info, point):

    # Read the CCMC data
    cdbfilepath = info['deltaB_files'][point]
    dBMhd1, dBFac1, dBHal, dBPed1 = extract_from_swmf_ccmc_printout_file( cdbfilepath )

    # Read the magnetopost data
    bs_msph1, bs_fac1, bs_hall, bs_pedersen1 = extract_from_magnetopost_files(info, point)
    
    # Magnetopost code has a minus sign error
    bs_hall['north'] = -bs_hall['north']
    bs_hall['east'] = -bs_hall['east']
    bs_hall['down'] = -bs_hall['down']

    # Read deltaB data
    pklname = 'dB_bs_iono-' + point + '.pkl'
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )

    # Calculate the magnitude of the B fields from the three methods
    df['mag']      = np.sqrt(df['Bnh']**2         + df['Beh']**2        + df['Bdh']**2)
    dBHal['mag']   = np.sqrt(dBHal['north']**2   + dBHal['east']**2   + dBHal['down']**2)
    bs_hall['mag']  = np.sqrt(bs_hall['north']**2  + bs_hall['east']**2  + bs_hall['down']**2)

    # Combine the two dataframes into one.  
    # NOTE, all the dataframes use the file timestamps to index the entries, 
    dBHal.columns =['Bn_cdb','Be_cdb','Bd_cdb','mag_cdb']        # CCMC
    bs_hall.columns =['Bn_mp','Be_mp','Bd_mp','mag_mp']      # magnetopost
    df_all = dBHal.merge( df, left_index=True, right_index=True) # deltaB
    df_all = df_all.merge( bs_hall, left_index=True, right_index=True) 
    
    # Calculate differences between the B fields from the three methods.
    df_all['cdb-db north'] = df_all['Bn_cdb'] - df_all['Bnh']
    df_all['cdb-db east'] = df_all['Be_cdb'] - df_all['Beh']
    df_all['cdb-db down'] = df_all['Bd_cdb'] - df_all['Bdh'] 
    df_all['cdb-db mag'] = df_all['mag_cdb'] - df_all['mag']
 
    df_all['mp-db north'] = df_all['Bn_mp'] - df_all['Bnh']
    df_all['mp-db east'] = df_all['Be_mp'] - df_all['Beh']
    df_all['mp-db down'] = df_all['Bd_mp'] - df_all['Bdh'] 
    df_all['mp-db mag'] = df_all['mag_mp'] - df_all['mag']

    # Plot results
    ax = df_all.plot(y='Bn_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bn_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bnh', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Hall $B_N$' )
     
    ax = df_all.plot(y='Be_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Be_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Beh', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Hall $B_E$' )
    
    ax = df_all.plot(y='Bd_cdb', label='CCMC')
    df_all.plot( ax=ax, y='Bd_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='Bdh', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Hall $B_D$' )
 
    ax = df_all.plot(y='mag_cdb', label='CCMC')
    df_all.plot( ax=ax, y='mag_mp', label='Magnetopost', ls='dashed')
    df_all.plot( ax=ax, y='mag', label='deltaB', ls='dotted')
    ax.set_ylabel( r'Hall $|B|$' )

    # ax = df_all.plot(y='cdb-db north', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db north', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Hall $\Delta B_N$' )

    # ax = df_all.plot(y='cdb-db east', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db east', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Hall $\Delta B_E$' )

    # ax = df_all.plot(y='cdb-db down', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db down', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Hall $\Delta B_D$' )

    # ax = df_all.plot(y='cdb-db mag', label='CCMC-deltaB')
    # df_all.plot( ax=ax, y='mp-db mag', label='Magnetopost-deltaB', ls='dashed')
    # ax.set_ylabel( r'Hall $\Delta |B|$' )

    print('Mean CCMC-deltaB Hall Bn: ', df_all['cdb-db north'].mean())
    print('Mean CCMC-deltaB Hall Be: ', df_all['cdb-db east'].mean())
    print('Mean CCMC-deltaB Hall Bd: ', df_all['cdb-db down'].mean())
    print('Mean CCMC-deltaB Hall |B|: ', df_all['cdb-db down'].mean())
    print('Std Dev CCMC-deltaB Hall Bn: ', df_all['cdb-db north'].std())
    print('Std Dev CCMC-deltaB Hall Be: ', df_all['cdb-db east'].std())
    print('Std Dev CCMC-deltaB Hall Bd: ', df_all['cdb-db down'].std())
    print('Std Dev CCMC-deltaB Hall |B|: ', df_all['cdb-db mag'].std())

    print('Mean Magnetopost-deltaB Hall Bn: ', df_all['mp-db north'].mean())
    print('Mean Magnetopost-deltaB Hall Be: ', df_all['mp-db east'].mean())
    print('Mean Magnetopost-deltaB Hall Bd: ', df_all['mp-db down'].mean())
    print('Mean Magnetopost-deltaB Hall |B|: ', df_all['mp-db down'].mean())
    print('Std Dev Magnetopost-deltaB Hall Bn: ', df_all['mp-db north'].std())
    print('Std Dev Magnetopost-deltaB Hall Be: ', df_all['mp-db east'].std())
    print('Std Dev Magnetopost-deltaB Hall Bd: ', df_all['mp-db down'].std())
    print('Std Dev Magnetopost-deltaB Hall |B|: ', df_all['mp-db mag'].std())
    return

if __name__ == "__main__":

    from SWPC_SWMF_052811_2_info import info

    from magnetopost import util as util
    util.setup(info)
    
    # Locations to compute B. See config.py for list of known points.
    point  = 'YKC'
    
    # Plot the results for the magnetosphere
    ms_plots(info, point)
    gap_plots(info, point)
    pedersen_plots(info, point)
    hall_plots(info, point)

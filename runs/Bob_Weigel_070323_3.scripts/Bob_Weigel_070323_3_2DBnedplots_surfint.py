#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:38:00 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import magnetopost as mp
from datetime import datetime


#################################################################
#
# Script for generating 2D Bn (B north) versus time,
# Be (B east) versus time, and Bd (B down) versus time plots.
# Results are provided for the magnetosphere based on BATS-R-US data
# and for the gap region and the ionosphere based on RIM data  
#
#################################################################

from Bob_Weigel_070323_3_info import info as info

# COMPUTE, True or False compute delta B contributions
# If false, only generate plots
COMPUTE=False

def extract_from_magnetopost_files(info, surface_location):

    msph_times = info['files']['magnetosphere'].keys()
    # iono_times = info['files']['ionosphere'].keys()

    msph_dtimes = [datetime(*time) for time in msph_times]
    # iono_dtimes = [datetime(*time) for time in iono_times]

    def get(ftag, dtimes):
        df = pd.DataFrame()
        infile = os.path.join( info['dir_derived'], 'timeseries', f'{ftag}-{surface_location}.npy' )
        mp.logger.info(f"Reading {infile}")
        dB = np.load(infile)

        df['north'] = pd.Series(data=dB[:,0], index=dtimes)
        df['east']  = pd.Series(data=dB[:,1], index=dtimes)
        df['down']  = pd.Series(data=dB[:,2], index=dtimes)
        
        Btimes = np.zeros(len(dtimes))
        for j in range(len(dtimes)): 
            dtime = dtimes[j]
            h = dtime.hour
            m = dtime.minute
            Btimes[j] = h + m/60
            
        df['Time'] = pd.Series(data=Btimes, index=dtimes)

        return df

    bs_msph     = get('bs_msph', msph_dtimes)
    bs_fac      = get('bs_fac', msph_dtimes)

    # bs_hall     = get('bs_hall', iono_dtimes)
    # bs_pedersen = get('bs_pedersen', iono_dtimes)
    
    helm_rCurr  = get( 'helm_rCurrents_gapSM', msph_dtimes )
    helm_outer  = get( 'helm_outer', msph_dtimes )
    cl_msph     = get( 'cl_msph', msph_dtimes )

    # return bs_msph, bs_fac, bs_hall, bs_pedersen
    return bs_msph, bs_fac, helm_rCurr, helm_outer, cl_msph

if __name__ == "__main__":

    # from magnetopost import util as util
    # util.setup(info)
    db.setup(info)
    
    # Location to compute B. See config.py for list of known points.
    point = "Colaba"
    
    # Do we skip files to save time.  If None, do all files.  If not
    # None, then reduce is an integer that determine how many files are skipped
    # e.g., do every 10th file
    reduce = None
    
    # Calculate the delta B sums to get Bn, Be,and Bd contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere.  Bn, Be, and Bd calcuated at points[0]
    if COMPUTE:
        db.loop_ms_b(info, point, reduce, maxcores=20)    
        # db.loop_gap_b(info, point, reduce, nR=100, useRIM=True)
        # db.loop_iono_b(info, point, reduce)
        db.loop_ms_surfint_rCurrents_b(info, point, reduce, maxcores=20, deltaBlist=False)    
        db.loop_ms_surfint_outer_b(info, point, reduce, maxcores=20, deltaBlist=False)    
        db.loop_ms_divBint_b(info, point, reduce, maxcores=20, deltaBlist=False)

    # # Plot the results
    # db.plot_Bned_ms_gap_iono(info, point)

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [12,4]
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 10
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    # Read the magnetopost data: magnetosphere, fac, inner and outer surface integrals, divB integral
    bs_msph, bs_fac, helm_rCurr, helm_outer, cl_msph = extract_from_magnetopost_files(info, point)
    
    # Rename the magnetopost columns for tidy names in plots
    bs_msph.columns = ['$B_N$ Biot Gary', r'$B_E$ Biot Gary', r'$B_D$ Biot Gary', r'Time (hr)']
    helm_rCurr.columns = ['$B_NG$', r'$B_EG$', r'$B_DG$', r'Time (hr)']
    helm_outer.columns = ['$B_NG$', r'$B_EG$', r'$B_DG$', r'Time (hr)']
    cl_msph.columns = ['$B_NG$', r'$B_EG$', r'$B_DG$', r'Time (hr)']
    
    # Read the deltaB magnetosphere, inner and outer surface integrals, and divB integral
    # Rename columns to make tidy names on plots
    pklname = 'dB_bs_msph-' + point + '.pkl'
    df_bs = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_bs.columns = [r'$B_N$ Biot-Savart', r'$B_E$ Biot-Savart', r'$B_D$ Biot-Savart', \
                      r'$B_{N\parallel}$', r'$B_{E\parallel}$', r'$B_{D\parallel}$', \
                      r'$B_{N\perp}$', r'$B_{E\perp}$', r'$B_{D\perp}$', \
                      r'$B_{N\perp\phi}$', r'$B_{E\perp\phi}$', r'$B_{D\perp\phi}$', \
                      r'$B_{N\Delta\perp}$', r'$B_{E\Delta\perp}$', r'$B_{D\Delta\perp}$', \
                      r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    pklname = 'dB_si_msph_rCurrents-' + point + '.pkl'
    df_si_inner = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_si_inner.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']

    pklname = 'dB_si_msph_outer-' + point + '.pkl'
    df_si_outer = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_si_outer.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    pklname = 'dB_divB_msph_b-' + point + '.pkl'
    df_divB = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_divB.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
       
    # pklname = 'dB_divB_msph_swmfio_b1-' + point + '.pkl'
    # df_divB3 = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    # df_divB3.columns = [r'$B_NFT$', r'$B_EFT$', r'$B_DFT$', r'$B_x$', r'$B_y$', r'$B_z$', \
    #              r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
       
    # pklname = 'dB_divB_msph_swmfio_b-' + point + '.pkl'
    # df_divB4 = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    # df_divB4.columns = [r'$B_NFF$', r'$B_EFF$', r'$B_DFF$', r'$B_x$', r'$B_y$', r'$B_z$', \
    #              r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap regiomn
    df_bs[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$'] - df_divB[r'$B_N$'] 
    df_bs[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$'] - df_divB[r'$B_E$'] 
    df_bs[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$'] - df_divB[r'$B_D$'] 
        
    df_bs[r'$B_N$ Inner + Outer'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$']
    df_bs[r'$B_E$ Inner + Outer'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$']  
    df_bs[r'$B_D$ Inner + Outer'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$']  
        
    df_bs[r'$B_N$ Inner'] = df_si_inner[r'$B_N$'] 
    df_bs[r'$B_E$ Inner'] = df_si_inner[r'$B_E$'] 
    df_bs[r'$B_D$ Inner'] = df_si_inner[r'$B_D$'] 
        
    # Create directory for plots
    db.create_directory( info['dir_plots'], 'BnedSurfInt'  )
    
    # Create plots and save them 

    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(df_bs[r'Time (hr)'], df_bs[r'$B_N$ Biot-Savart'],'k-', label=r'$B_{BS}$' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point)
    ax[0].set_xlabel('Time (UTC)')
    l2 = ax[0].plot(df_bs[r'Time (hr)'], df_bs[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:', 
               label=r'$B_{H} $+ $\delta B_{outer}$ + $\delta B_{div}$' )
    l3 = ax[0].plot(df_bs[r'Time (hr)'], df_bs[r'$B_N$ Inner + Outer'], 'g-', label=r'$B_{H}$ + $\delta B_{outer}$ ')
    l4 = ax[0].plot(df_bs[r'Time (hr)'], df_bs[r'$B_N$ Inner'], 'b-', label=r'$B_{H}$' )
    ax[0].set_xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$',
                  r'$\mathsf{B}_{\mathsf{HDT}}+\mathsf{\delta B_{out}}+\mathsf{\delta B_{div}}$',
                  r'$\mathsf{B}_{\mathsf{HDT}}+\mathsf{\delta B_{out}}$',
                  r'$\mathsf{B}_{\mathsf{HDT}}$'])
    # ax[0].legend()
    
    ax[1].plot(df_bs[r'Time (hr)'], df_bs[r'$B_E$ Biot-Savart'],'k-' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].plot(df_bs[r'Time (hr)'], df_bs[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:')
    ax[1].plot(df_bs[r'Time (hr)'], df_bs[r'$B_E$ Inner + Outer'], 'g-' )
    ax[1].plot(df_bs[r'Time (hr)'], df_bs[r'$B_E$ Inner'], 'b-' )
    ax[1].set_xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    
    ax[2].plot(df_bs[r'Time (hr)'], df_bs[r'$B_D$ Biot-Savart'],'k-' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].plot(df_bs[r'Time (hr)'], df_bs[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:')
    ax[2].plot(df_bs[r'Time (hr)'], df_bs[r'$B_D$ Inner + Outer'], 'g-' )
    ax[2].plot(df_bs[r'Time (hr)'], df_bs[r'$B_D$ Inner'], 'b-' )
    ax[2].set_xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    
    plt.tight_layout()
    
    pltname = 'tot-Bned-Test-' + point
    fig.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )


     

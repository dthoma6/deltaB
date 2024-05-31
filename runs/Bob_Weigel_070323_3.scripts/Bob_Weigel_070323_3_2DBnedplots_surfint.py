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
from copy import deepcopy
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
    iono_times = info['files']['ionosphere'].keys()

    msph_dtimes = [datetime(*time) for time in msph_times]
    iono_dtimes = [datetime(*time) for time in iono_times]

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

    from magnetopost import util as util
    util.setup(info)
    
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
        # db.loop_ms_b(info, point, reduce)    
        # db.loop_gap_b(info, point, reduce, nR=100, useRIM=True)
        # db.loop_iono_b(info, point, reduce)
        db.loop_ms_surfint_rCurrents_b(info, point, reduce, maxcores=20, deltaBlist=False)    
        db.loop_ms_surfint_outer_b(info, point, reduce, maxcores=20, deltaBlist=False)    
        db.loop_ms_divBint_b(info, point, reduce, maxcores=20, deltaBlist=False)

    # # Plot the results
    # db.plot_Bned_ms_gap_iono(info, point)

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [6,4.5]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        "text.usetex": True,
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
    
    pklname = 'dB_divB_msph_2nd_b1-' + point + '.pkl'
    df_divB = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    df_divB.columns = [r'$B_NTT$', r'$B_ETT$', r'$B_DTT$', r'$B_x$', r'$B_y$', r'$B_z$', \
                 r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
       
    # pklname = 'dB_divB_msph_2nd_b-' + point + '.pkl'
    # df_divB2 = pd.read_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    # df_divB2.columns = [r'$B_NTF$', r'$B_ETF$', r'$B_DTF$', r'$B_x$', r'$B_y$', r'$B_z$', \
    #              r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
       
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
    df_bs[r'$B_N$ Surf Ints + divB'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$'] - df_divB[r'$B_NTT$'] 
    df_bs[r'$B_E$ Surf Ints + divB'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$'] - df_divB[r'$B_ETT$'] 
    df_bs[r'$B_D$ Surf Ints + divB'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$'] - df_divB[r'$B_DTT$'] 
        
    df_bs[r'$B_N$ Surf Ints Only'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$']
    df_bs[r'$B_E$ Surf Ints Only'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$']  
    df_bs[r'$B_D$ Surf Ints Only'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$']  
        
    # Create directory for plots
    db.create_directory( info['dir_plots'], 'BnedSurfInt'  )
    
    # Create plots and save them 

    ax = df_si_inner.plot.line(x=r'Time (hr)', y=[r'$B_N$'],\
                style=['-'], \
                grid = False,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Bob_Weigel_070323_3 at ' + point)
    helm_rCurr.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bn-surfint-rCurrents-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_inner.plot(x=r'Time (hr)', y=[r'$B_E$'],\
                style=['-'], \
                grid = False,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Bob_Weigel_070323_3 at ' + point)
    helm_rCurr.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_EG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Be-surfint-rCurrents-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_inner.plot(x=r'Time (hr)', y=[r'$B_D$'],\
                style=['-'], \
                grid = False,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Bob_Weigel_070323_3 at ' + point)
    helm_rCurr.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bd-surfint-rCurrents-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_outer.plot.line(x=r'Time (hr)', y=[r'$B_N$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Bob_Weigel_070323_3 at ' + point)
    helm_outer.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bn-surfint-outer-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_outer.plot(x=r'Time (hr)', y=[r'$B_E$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Bob_Weigel_070323_3 at ' + point)
    helm_outer.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_EG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Be-surfint-outer-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_outer.plot(x=r'Time (hr)', y=[r'$B_D$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Bob_Weigel_070323_3 at ' + point)
    helm_outer.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bd-surfint-outer-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot.line(x=r'Time (hr)', y=[r'$B_NTT$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_NTT$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Bob_Weigel_070323_3 at ' + point)
    # df_divB2.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NTF$'],\
    #             style=['-.'] )
    # df_divB3.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NFT$'],\
    #             style=['-.'] )
    # df_divB4.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NFF$'],\
    #             style=['-.'] )
    cl_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_NG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bn-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot(x=r'Time (hr)', y=[r'$B_ETT$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_ETT$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Bob_Weigel_070323_3 at ' + point)
    # df_divB2.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_ETF$'],\
    #             style=['-.'] )
    # df_divB3.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_EFT$'],\
    #             style=['-.'] )
    # df_divB4.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_EFF$'],\
    #             style=['-.'] )
    cl_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_EG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Be-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot(x=r'Time (hr)', y=[r'$B_DTT$'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_DTT$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Bob_Weigel_070323_3 at ' + point)
    # df_divB2.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DTF$'],\
    #             style=['-.'] )
    # df_divB3.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DFT$'],\
    #             style=['-.'] )
    # df_divB4.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DFF$'],\
    #             style=['-.'] )
    cl_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_DG$'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bd-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

   ####
   
    ax = df_bs.plot.line(x=r'Time (hr)', y=[r'$B_N$ Biot-Savart'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Bob_Weigel_070323_3 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_N$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_N$ Surf Ints Only'],\
                style=['--'] )
    # bs_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_N$ Biot Gary'],\
    #             style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bn-Test-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_bs.plot(x=r'Time (hr)', y=[r'$B_E$ Biot-Savart'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Bob_Weigel_070323_3 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_E$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_E$ Surf Ints Only'],\
                style=['--'] )
    # bs_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_E$ Biot Gary'],\
    #             style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Be-Test-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_bs.plot(x=r'Time (hr)', y=[r'$B_D$ Biot-Savart'],\
                legend=True, \
                style=['-'], \
                grid = False,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Bob_Weigel_070323_3 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_D$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_D$ Surf Ints Only'],\
                style=['--'] )
    # bs_msph.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_D$ Biot Gary'],\
    #             style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    # plt.show()
    pltname = 'tot-Bd-Test-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )


     

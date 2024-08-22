#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:38:51 2024

@author: Dean Thomas
"""
import os.path
import deltaB as db
import pandas as pd
import matplotlib.pyplot as plt

from Dean_Thomas_052924_1_info import info as info

# COMPUTE, True or False compute delta B contributions
# If false, only generate plots
COMPUTE=True

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
           
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap region
    df_bs[r'$B_N$ Surf Ints + divB'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$'] - df_divB[r'$B_N$'] 
    df_bs[r'$B_E$ Surf Ints + divB'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$'] - df_divB[r'$B_E$'] 
    df_bs[r'$B_D$ Surf Ints + divB'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$'] - df_divB[r'$B_D$'] 
        
    df_bs[r'$B_N$ Surf Ints Only'] = df_si_inner[r'$B_N$'] - df_si_outer[r'$B_N$']
    df_bs[r'$B_E$ Surf Ints Only'] = df_si_inner[r'$B_E$'] - df_si_outer[r'$B_E$']  
    df_bs[r'$B_D$ Surf Ints Only'] = df_si_inner[r'$B_D$'] - df_si_outer[r'$B_D$']  
        
    # Create directory for plots
    db.create_directory( info['dir_plots'], 'BnedSurfInt'  )
    
    # Create plots and save them 

    ax = df_si_inner.plot.line(x=r'Time (hr)', y=[r'$B_N$'],\
                style=['-'], \
                grid = True,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Bn-surfint-rCurrents-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_inner.plot(x=r'Time (hr)', y=[r'$B_E$'],\
                style=['-'], \
                grid = True,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Be-surfint-rCurrents-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_si_inner.plot(x=r'Time (hr)', y=[r'$B_D$'],\
                style=['-'], \
                grid = True,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Inner Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
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
                grid = True,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
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
                grid = True,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
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
                grid = True,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Outer Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Bd-surfint-outer-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot.line(x=r'Time (hr)', y=[r'$B_N$'],\
                legend=True, \
                style=['-'], \
                grid = True,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Bn-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot(x=r'Time (hr)', y=[r'$B_E$'],\
                legend=True, \
                style=['-'], \
                grid = True,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Be-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

    ax = df_divB.plot(x=r'Time (hr)', y=[r'$B_D$'],\
                legend=True, \
                style=['-'], \
                grid = True,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'divB Dean_Thomas_052924_1 at ' + point)
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Bd-divB-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

   # ####
   
    ax = df_bs.plot.line(x=r'Time (hr)', y=[r'$B_N$ Biot-Savart'],\
                legend=True, \
                style=['-'], \
                grid = True,\
                ylabel = r'$B_N$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Dean_Thomas_052924_1 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_N$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_N$ Surf Ints Only'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
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
                grid = True,\
                ylabel = r'$B_E$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Dean_Thomas_052924_1 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_E$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_E$ Surf Ints Only'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
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
                grid = True,\
                ylabel = r'$B_D$ at ' + point, xlabel = 'Time (UTC)', \
                title = 'Test Dean_Thomas_052924_1 at ' + point)
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_D$ Surf Ints + divB'],\
                style=['--'] )
    df_bs.plot.line(ax=ax, x=r'Time (hr)', y=[r'$B_D$ Surf Ints Only'],\
                style=['--'] )
    plt.xticks(ticks=[1,4,7,10,13,16,19],labels=['01:00', '04:00', '07:00', '10:00', '13:00', '16:00', '19:00']) 
    pltname = 'tot-Bd-Test-' + point
    plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )


    

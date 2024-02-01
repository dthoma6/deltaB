#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import os.path
from deltaB import loop_2D_ms, plot_2D_ms, \
    loop_2D_gap_iono, plot_2D_gap_iono, \
    plot_2D_ms_gap_iono, date_timeISO, \
    loop_2D_ms_point, loop_2D_gap_iono_point, \
    create_directory

from Chigomezyo_Ngwira_092112_3a_info import info as info
from Chigomezyo_Ngwira_092112_3a_info import setup as setup

if __name__ == "__main__":
    
    # Shift the results by DELTAHR hours.
    DELTAHR = 5.5
    
    # Point on Earth surface where B is measured
    POINT = 'Colaba'
    
    # Limits of time (x) and Bn (y) axes in plots
    TIME_LIMITS = [0,4]
    BN_LIMITS = [-1200,400]

    # If None, process all files.  If not None, its an integer that says 
    # we're skipping every n files.
    REDUCE = None

    # Get a list of BATSRUS and RIM files. info parameters define location 
    # (dir_run) and file types.  Based on info structure from magnetopost.
    # NOTE: magnetopost code fails on this, so a modified version of setup
    # is in Chigomezyo_Ngwira_092112_3a_info
    setup(info)
    
    # Calculate the delta B sums to get Bn contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere at POINT
    loop_2D_ms_point(POINT, info, REDUCE, deltahr=DELTAHR)
    loop_2D_gap_iono_point(POINT, info, REDUCE, nR=100, deltahr=DELTAHR, useRIM=True)
 
    # Create 2d plots of Bn vs. time
    # plot_2D_ms( POINT, info, TIME_LIMITS, BN_LIMITS )
    # plot_2D_gap_iono( POINT, info, TIME_LIMITS, BN_LIMITS )
    # plot_2D_ms_gap_iono( POINT, info, TIME_LIMITS, BN_LIMITS )

    # Customized plots for paper
    
    import matplotlib.pyplot as plt
    import pandas as pd

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

    # Read data gnerated above
    pklname = info['run_name'] + '.ms-2D-' + POINT +'.pkl'
    df1 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    pklname = info['run_name'] + '.gap-iono-2D-' + POINT +'.pkl'
    df2 = pd.read_pickle( os.path.join( info['dir_derived'], '2D', pklname) )

    df1.columns =['MS Total',r'$j_{\parallel}$',r'$j_\perp$', r'$j_{\perp \phi}$', \
                r'$\Delta j_\perp$', 'Time', 'Datetime', 'Month', \
                'Day', 'Hour', 'Minute' ]

    ax = df1.plot( x=r'Time', y=['MS Total'], \
                    xlabel=r'Time (UTC)', \
                    ylabel=r'$B_{N}$ (nT) at ' + POINT, \
                    style=['-','r:','--'], \
                    grid = False,\
                    # legend=True) 
                    legend=False, title=r'$B_{N}$ due to Total Magnetospheric Currents') 
    plt.yticks(ticks=[600,400,200,0,-200,-400,-600,-800,-1000,-1200,-1400], labels=['600','400','200','0','-200','-400','-600','-800','-1000','-1200','-1400'])
    plt.xticks(ticks=[5,6,7,8,9],labels=['05:00', '06:00', '07:00', '08:00', '09:00']) 
    plt.axvline(x=0.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=1.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=2.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=3.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=4.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.tight_layout()
    
    # Create directory for plots
    create_directory( info['dir_plots'], 'currents-regions'  )
    
    # Save plot
    plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-total.svg' ) )

    ax = df1.plot( x=r'Time', y=[r'$j_{\parallel}$', r'$j_{\perp \phi}$', \
                    r'$\Delta j_\perp$'], \
                    xlabel=r'Time (UTC)', \
                    ylabel=r'$B_{N}$ (nT) at ' + POINT, \
                    style=['-','r:','--'], \
                    grid = False,\
                    # legend=True) 
                    legend=True, title=r'$B_{N}$ due to Magnetospheric Currents') 
    plt.yticks(ticks=[200,0,-200,-400,-600,-800,-1000,-1200,-1400,-1600,-1800,-2000], labels=['200','0','-200','-400','-600','-800','-1000','-1200','-1400','-1600','-1800','-2000'])
    plt.xticks(ticks=[5,6,7,8,9],labels=['05:00', '06:00', '07:00', '08:00', '09:00']) 
    plt.axvline(x=0.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=1.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=2.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=3.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=4.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.tight_layout()
   
    # Save plot
    plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'ms-components.svg' ) )

    df2.columns =['Gap $j_{\parallel}$', '$j_P$', '$j_H$', 'Time', \
                    'Datetime', 'Month', 'Day', 'Hour', 'Minute'  ]

    ax = df2.plot( x=r'Time', y=[r'Gap $j_{\parallel}$', r'$j_P$', r'$j_H$'], \
                    xlabel=r'Time (UTC)', \
                    ylabel=r'$B_{N}$ (nT) at ' + POINT, \
                    style=['-','r:','--'], \
                    grid = False,\
                    # legend=True)   
                    legend=True, title=r'$B_{N}$ due to Gap and Ionospheric Currents')   
    plt.yticks(ticks=[600,400,200,0,-200,-400,-600,-800], labels=['600','400','200','0','-200','-400','-600','-800'])
    plt.xticks(ticks=[5,6,7,8,9],labels=['05:00', '06:00', '07:00', '08:00', '09:00']) 
    plt.axvline(x=0.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=1.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=2.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=3.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.axvline(x=4.0+DELTAHR, ls='dotted', c='green', alpha=0.95)    
    plt.tight_layout()
 
    # Save plot
    plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'currents-regions', 'gap-iono.svg' ) )


   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:08:36 2024

@author: Dean Thomas
"""

import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory
from datetime import datetime
import numpy as np

from Bob_Weigel_070323_3_info import info as bwinfo

point = 'Colaba'

# # ext = '_bz_flip'
# # ext = '_line_current'
# ext = '_no_transforms'

# exts =['_bz_flip', '_line_current', '_no_transforms', '_bz_flip_gt_pi']
exts =['_bz_flip_gt_pi', ]
labels = [r' ($\mathsf{B}_{\mathsf{Z}}$ flip)', ' (Line Current)', '']

bwroot = '/Volumes/PhysicsHD/Bob_Weigel_070323_3.derived/timeseries'
dtroot = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived/timeseries'

bwdata_dir = r'/Volumes/PhysicsHD'
dtdata_dir = r'/Volumes/PhysicsHD'

bwinfo = {
        "model": "SWMF",
        "run_name": "Bob_Weigel_070323_3",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "method": "method1",
        "dir_run": os.path.join(dtdata_dir, "Bob_Weigel_070323_3"),
        "dir_plots": os.path.join(dtdata_dir, "Bob_Weigel_070323_3.plots"),
        "dir_derived": os.path.join(dtdata_dir, "Bob_Weigel_070323_3.derived"),
        "dir_magnetosphere": os.path.join(dtdata_dir, "Bob_Weigel_070323_3", "GM_CDF"),
        "dir_ionosphere": os.path.join(dtdata_dir, "Bob_Weigel_070323_3", "IONO-2D_CDF")
}
dtinfo = {
        "model": "OpenGGCM",
        "run_name": "Dean_Thomas_052924_1",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(dtdata_dir, "Dean_Thomas_052924_1"),
        "dir_plots": os.path.join(dtdata_dir, "Dean_Thomas_052924_1.plots"),
        "dir_derived": os.path.join(dtdata_dir, "Dean_Thomas_052924_1.derived"),
        "dir_magnetosphere": os.path.join(dtdata_dir, "Dean_Thomas_052924_1", "GM_CDF"),
        "dir_ionosphere": os.path.join(dtdata_dir, "Dean_Thomas_052924_1", "IONO-2D_CDF")
}

for ii in range(len(exts)):
    
    ext = exts[ii]
    label = labels[ii]
    
    bwrootdir = bwroot + ext
    dtrootdir = dtroot + ext
        
    # Create diretory for plots
    create_directory( bwinfo['dir_plots'], 'BnedSurfInt'+ext )
    create_directory( dtinfo['dir_plots'], 'BnedSurfInt'+ext )
    
    # Read pickle files
    biot = 'dB_bs_msph-Colaba.pkl'
    bwbiotdf = pd.read_pickle( os.path.join( bwrootdir, biot ) )
    dtbiotdf = pd.read_pickle( os.path.join( dtrootdir, biot ) )
    
    inner = 'dB_si_msph_rCurrents-Colaba.pkl'
    bwinnerdf = pd.read_pickle( os.path.join( bwrootdir, inner ) )
    dtinnerdf = pd.read_pickle( os.path.join( dtrootdir, inner ) )
    
    outer = 'dB_si_msph_outer-Colaba.pkl'
    bwouterdf = pd.read_pickle( os.path.join( bwrootdir, outer ) )
    dtouterdf = pd.read_pickle( os.path.join( dtrootdir, outer ) )
    
    divb = 'dB_divB_msph_b-Colaba.pkl'
    bwdivbdf = pd.read_pickle( os.path.join( bwrootdir, divb ) )
    dtdivbdf = pd.read_pickle( os.path.join( dtrootdir, divb ) )
    
    # Rename columns
    bwbiotdf.columns = [r'$B_N$ Biot-Savart', r'$B_E$ Biot-Savart', r'$B_D$ Biot-Savart', \
                      r'$B_{N\parallel}$', r'$B_{E\parallel}$', r'$B_{D\parallel}$', \
                      r'$B_{N\perp}$', r'$B_{E\perp}$', r'$B_{D\perp}$', \
                      r'$B_{N\perp\phi}$', r'$B_{E\perp\phi}$', r'$B_{D\perp\phi}$', \
                      r'$B_{N\Delta\perp}$', r'$B_{E\Delta\perp}$', r'$B_{D\Delta\perp}$', \
                      r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    dtbiotdf.columns = [r'$B_N$ Biot-Savart', r'$B_E$ Biot-Savart', r'$B_D$ Biot-Savart', \
                      r'$B_{N\parallel}$', r'$B_{E\parallel}$', r'$B_{D\parallel}$', \
                      r'$B_{N\perp}$', r'$B_{E\perp}$', r'$B_{D\perp}$', \
                      r'$B_{N\perp\phi}$', r'$B_{E\perp\phi}$', r'$B_{D\perp\phi}$', \
                      r'$B_{N\Delta\perp}$', r'$B_{E\Delta\perp}$', r'$B_{D\Delta\perp}$', \
                      r'$B_x$', r'$B_y$', r'$B_z$', \
                      r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    bwinnerdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    dtinnerdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    bwouterdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    dtouterdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_{Nirr}$', r'$B_{Eirr}$', r'$B_{Dirr}$', \
                  r'$B_{Nsol}$', r'$B_{Esol}$', r'$B_{Dsol}$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    bwdivbdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    dtdivbdf.columns = [r'$B_N$', r'$B_E$', r'$B_D$', r'$B_x$', r'$B_y$', r'$B_z$', \
                  r'Time (hr)', r'Datetime', r'Month', r'Day', r'Hour', r'Minute']
    
    ######################################################################################
    ######################################################################################
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap region
    ######################################################################################
    ######################################################################################
    
    bwbiotdf[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = bwinnerdf[r'$B_N$'] - bwouterdf[r'$B_N$'] - bwdivbdf[r'$B_N$'] 
    bwbiotdf[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = bwinnerdf[r'$B_E$'] - bwouterdf[r'$B_E$'] - bwdivbdf[r'$B_E$'] 
    bwbiotdf[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = bwinnerdf[r'$B_D$'] - bwouterdf[r'$B_D$'] - bwdivbdf[r'$B_D$'] 
        
    bwbiotdf[r'$B_N$ Inner + Outer'] = bwinnerdf[r'$B_N$'] - bwouterdf[r'$B_N$']
    bwbiotdf[r'$B_E$ Inner + Outer'] = bwinnerdf[r'$B_E$'] - bwouterdf[r'$B_E$']  
    bwbiotdf[r'$B_D$ Inner + Outer'] = bwinnerdf[r'$B_D$'] - bwouterdf[r'$B_D$']  
        
    bwbiotdf[r'$B_N$ Outer'] = - bwouterdf[r'$B_N$']
    bwbiotdf[r'$B_E$ Outer'] = - bwouterdf[r'$B_E$']  
    bwbiotdf[r'$B_D$ Outer'] = - bwouterdf[r'$B_D$']  
        
    bwbiotdf[r'$B_N$ Inner'] = bwinnerdf[r'$B_N$'] 
    bwbiotdf[r'$B_E$ Inner'] = bwinnerdf[r'$B_E$'] 
    bwbiotdf[r'$B_D$ Inner'] = bwinnerdf[r'$B_D$'] 
    
    bwbiotdf[r'$B_N$ divB'] = - bwdivbdf[r'$B_N$'] 
    bwbiotdf[r'$B_E$ divB'] = - bwdivbdf[r'$B_E$'] 
    bwbiotdf[r'$B_D$ divB'] = - bwdivbdf[r'$B_D$'] 
        
    dtbiotdf[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = dtinnerdf[r'$B_N$'] - dtouterdf[r'$B_N$'] - dtdivbdf[r'$B_N$'] 
    dtbiotdf[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = dtinnerdf[r'$B_E$'] - dtouterdf[r'$B_E$'] - dtdivbdf[r'$B_E$'] 
    dtbiotdf[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'] = dtinnerdf[r'$B_D$'] - dtouterdf[r'$B_D$'] - dtdivbdf[r'$B_D$'] 
        
    dtbiotdf[r'$B_N$ Inner + Outer'] = dtinnerdf[r'$B_N$'] - dtouterdf[r'$B_N$']
    dtbiotdf[r'$B_E$ Inner + Outer'] = dtinnerdf[r'$B_E$'] - dtouterdf[r'$B_E$']  
    dtbiotdf[r'$B_D$ Inner + Outer'] = dtinnerdf[r'$B_D$'] - dtouterdf[r'$B_D$']  
        
    dtbiotdf[r'$B_N$ Outer'] = - dtouterdf[r'$B_N$']
    dtbiotdf[r'$B_E$ Outer'] = - dtouterdf[r'$B_E$']  
    dtbiotdf[r'$B_D$ Outer'] = - dtouterdf[r'$B_D$']  
        
    dtbiotdf[r'$B_N$ Inner'] = dtinnerdf[r'$B_N$'] 
    dtbiotdf[r'$B_E$ Inner'] = dtinnerdf[r'$B_E$'] 
    dtbiotdf[r'$B_D$ Inner'] = dtinnerdf[r'$B_D$'] 
    
    dtbiotdf[r'$B_N$ divB'] = - dtdivbdf[r'$B_N$'] 
    dtbiotdf[r'$B_E$ divB'] = - dtdivbdf[r'$B_E$'] 
    dtbiotdf[r'$B_D$ divB'] = - dtdivbdf[r'$B_D$'] 
        
    # Set some plot configs
    
    plt.rcParams["figure.figsize"] = [12,4]
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        # "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    
    # Create plots and save them 
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Biot-Savart'],'k-.', label=r'$\mathsf{B}_{\mathsf{BS}}$')
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:', 
               label=r'$\mathsf{B}_{\mathsf{in}}+\mathsf{B_{out}}+\mathsf{B_{div}}$' )
    l3 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Inner'], 'g:', label=r'$\mathsf{B}_{\mathsf{in}}$')
    l4 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Outer'], 'b:', label=r'$\mathsf{B}_{\mathsf{out}}$' )
    l5 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ divB'], 'm:', label=r'$\mathsf{B}_{\mathsf{div}}$' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$',
                  r'$\mathsf{B}_{\mathsf{in}}+\mathsf{B_{out}}+\mathsf{B_{div}}$',
                  r'$\mathsf{B}_{\mathsf{in}}$',
                  r'$\mathsf{B}_{\mathsf{out}}$',
                  r'$\mathsf{B}_{\mathsf{div}}$'])
    # ax[0].legend()
    
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Biot-Savart'],'k-.')
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Inner'], 'g:' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Outer'], 'b:' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ divB'], 'm:' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Biot-Savart'],'k-.')
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Inner'], 'g:' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Outer'], 'b:' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ divB'], 'm:' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'tot-Bned-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Biot-Savart'],'k-.', label=r'$\mathsf{B}_{\mathsf{BS}}$')
    l2 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:', 
               label=r'$\mathsf{B}_{\mathsf{in}}+\mathsf{B_{out}}+\mathsf{B_{div}}$' )
    l3 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Inner'], 'g:', label=r'$\mathsf{B}_{\mathsf{in}}$')
    l4 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Outer'], 'b:', label=r'$\mathsf{B}_{\mathsf{out}}$' )
    l5 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ divB'], 'm:', label=r'$\mathsf{B}_{\mathsf{div}}$' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$',
                  r'$\mathsf{B}_{\mathsf{in}}+\mathsf{B_{out}}+\mathsf{B_{div}}$',
                  r'$\mathsf{B}_{\mathsf{in}}$',
                  r'$\mathsf{B}_{\mathsf{out}}$',
                  r'$\mathsf{B}_{\mathsf{div}}$'])
    # ax[0].legend()
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Biot-Savart'],'k-.')
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Inner'], 'g:' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Outer'], 'b:' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ divB'], 'm:' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Biot-Savart'],'k-.')
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Inner + Outer + $\nabla \cdot \mathbf{B}$'], 'r:' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Inner'], 'g:' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Outer'], 'b:' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ divB'], 'm:' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'tot-Bned-Test-' + point
    fig.savefig( os.path.join( dtinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Biot-Savart'],'k-.' )
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Biot-Savart'],'r:')
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{BS}}$ SWMF'])
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Biot-Savart'],'k-.' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Biot-Savart'],'r:' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Biot-Savart'],'k-.')
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Biot-Savart'],'r:' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Compare-BBS-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Inner'],'k-.' )
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Inner'],'r:' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{in}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{in}}$ SWMF'])
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Inner'],'k-.')
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Inner'],'r:')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Inner'],'k-.')
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Inner'],'r:')
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Compare-Bin-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Outer'],'k-.' )
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Outer'],'r:' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{out}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{out}}$ SWMF'])
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Outer'],'k-.')
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Outer'],'r:')
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Outer'],'k-.')
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Outer'],'r:')
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Compare-Bout-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ divB'],'k-.' )
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ divB'],'r:' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{div}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{div}}$ SWMF'])
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ divB'],'k-.')
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ divB'],'r:')
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ divB'],'k-.')
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ divB'],'r:')
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Compare-Bdiv-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Biot-Savart'],'k-.' )
    l2 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ divB'],'r:' )
    l3 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Inner'],'g:' )
    l4 = ax[0].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_N$ Outer'],'b:' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$ SWMF',
                  r'$\mathsf{B}_{\mathsf{div}}$ SWMF',
                  r'$\mathsf{B}_{\mathsf{in}}$ SWMF',
                  r'$\mathsf{B}_{\mathsf{out}}$ SWMF',])
    
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Biot-Savart'],'k-.' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ divB'],'r:' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Inner'],'g:' )
    ax[1].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_E$ Outer'],'b:' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Biot-Savart'],'k-.' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ divB'],'r:' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Inner'],'g:' )
    ax[2].plot(bwbiotdf[r'Time (hr)'], bwbiotdf[r'$B_D$ Outer'],'b:' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Each-Bcomp-Test-' + point
    fig.savefig( os.path.join( bwinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    #######################
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    l1 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Biot-Savart'],'k-.' )
    l2 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ divB'],'r:' )
    l3 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Inner'],'g:' )
    l4 = ax[0].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_N$ Outer'],'b:' )
    ax[0].set_ylabel(r'$\mathsf{B_N}$ at ' + point + label)
    ax[0].set_xlabel('Time (UTC)')
    ax[0].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    ax[0].legend([r'$\mathsf{B}_{\mathsf{BS}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{div}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{in}}$ OpenGGCM',
                  r'$\mathsf{B}_{\mathsf{out}}$ OpenGGCM',])
    
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Biot-Savart'],'k-.' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ divB'],'r:' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Inner'],'g:' )
    ax[1].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_E$ Outer'],'b:' )
    ax[1].set_ylabel(r'$\mathsf{B_E}$ at ' + point + label)
    ax[1].set_xlabel('Time (UTC)')
    ax[1].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Biot-Savart'],'k-.' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ divB'],'r:' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Inner'],'g:' )
    ax[2].plot(dtbiotdf[r'Time (hr)'], dtbiotdf[r'$B_D$ Outer'],'b:' )
    ax[2].set_ylabel(r'$\mathsf{B_D}$ at ' + point + label)
    ax[2].set_xlabel('Time (UTC)')
    ax[2].set_xticks(ticks=[1,7,13,19],labels=['01:00', '07:00', '13:00', '19:00'])
    
    plt.tight_layout()
    plt.show()
    
    pltname = 'Each-Bcomp-Test-' + point
    fig.savefig( os.path.join( dtinfo['dir_plots'], 'BnedSurfInt'+ext, pltname + '.png' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
    # plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )
    
    ###########################
    
    # Some summary statistics to describe differences between SWMF and OpenGGCM results
    
    bsn = bwbiotdf[r'$B_N$ Biot-Savart'] - dtbiotdf[r'$B_N$ Biot-Savart']
    bse = bwbiotdf[r'$B_E$ Biot-Savart'] - dtbiotdf[r'$B_E$ Biot-Savart']
    bsd = bwbiotdf[r'$B_D$ Biot-Savart'] - dtbiotdf[r'$B_D$ Biot-Savart']

    inn = bwbiotdf[r'$B_N$ Inner'] - dtbiotdf[r'$B_N$ Inner']
    ine = bwbiotdf[r'$B_E$ Inner'] - dtbiotdf[r'$B_E$ Inner']
    ind = bwbiotdf[r'$B_D$ Inner'] - dtbiotdf[r'$B_D$ Inner']

    outn = bwbiotdf[r'$B_N$ Outer'] - dtbiotdf[r'$B_N$ Outer']
    oute = bwbiotdf[r'$B_E$ Outer'] - dtbiotdf[r'$B_E$ Outer']
    outd = bwbiotdf[r'$B_D$ Outer'] - dtbiotdf[r'$B_D$ Outer']

    divn = bwbiotdf[r'$B_N$ divB'] - dtbiotdf[r'$B_N$ divB']
    dive = bwbiotdf[r'$B_E$ divB'] - dtbiotdf[r'$B_E$ divB']
    divd = bwbiotdf[r'$B_D$ divB'] - dtbiotdf[r'$B_D$ divB']

    print()
    print( label )
    print( 'Biot-Savart N std: ', np.std(bsn))
    print( 'Biot-Savart E std: ', np.std(bse))
    print( 'Biot-Savart D std: ', np.std(bsd))
    print()
    print( 'Inner N std: ', np.std(inn))
    print( 'Inner E std: ', np.std(ine))
    print( 'Inner D std: ', np.std(ind))
    print()
    print( 'Outer N std: ', np.std(outn))
    print( 'Outer E std: ', np.std(oute))
    print( 'Outer D std: ', np.std(outd))
    print()
    print( 'divB N std: ', np.std(divn))
    print( 'divB E std: ', np.std(dive))
    print( 'divB D std: ', np.std(divd))
    
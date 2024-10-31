#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 07:16:51 2024

@author: Dean Thomas
"""

# Analyze size of divB, inner surface, and outer surface integral contributions
# to Biot-Savart

rootdir = '/Volumes/PhysicsHD/CARR_Scenario1.derived/heatmaps'

biotfiles = [ '3d__var_2_e20190902-050000-008.out.cdf.ms-heatmap-world.pkl',
              '3d__var_2_e20190902-060000-021.out.cdf.ms-heatmap-world.pkl',
              '3d__var_2_e20190902-063000-000.out.cdf.ms-heatmap-world.pkl',
              '3d__var_2_e20190902-070000-888.out.cdf.ms-heatmap-world.pkl',
              '3d__var_2_e20190902-080000-497.out.cdf.ms-heatmap-world.pkl' ]

divBfiles = [ '3d__var_2_e20190902-050000-008.out.cdf.divB-heatmap-world.pkl',
              '3d__var_2_e20190902-060000-021.out.cdf.divB-heatmap-world.pkl',
              '3d__var_2_e20190902-063000-000.out.cdf.divB-heatmap-world.pkl',
              '3d__var_2_e20190902-070000-888.out.cdf.divB-heatmap-world.pkl',
              '3d__var_2_e20190902-080000-497.out.cdf.divB-heatmap-world.pkl' ]

innerfiles = [ '3d__var_2_e20190902-050000-008.out.cdf.inner-heatmap-world.pkl',
              '3d__var_2_e20190902-060000-021.out.cdf.inner-heatmap-world.pkl',
              '3d__var_2_e20190902-063000-000.out.cdf.inner-heatmap-world.pkl',
              '3d__var_2_e20190902-070000-888.out.cdf.inner-heatmap-world.pkl',
              '3d__var_2_e20190902-080000-497.out.cdf.inner-heatmap-world.pkl'  ]

outerfiles = [ '3d__var_2_e20190902-050000-008.out.cdf.outer-heatmap-world.pkl',
              '3d__var_2_e20190902-060000-021.out.cdf.outer-heatmap-world.pkl',
              '3d__var_2_e20190902-063000-000.out.cdf.outer-heatmap-world.pkl',
              '3d__var_2_e20190902-070000-888.out.cdf.outer-heatmap-world.pkl',
              '3d__var_2_e20190902-080000-497.out.cdf.outer-heatmap-world.pkl' ]

times = [ '2009-09-02 05:00:00', 
         '2009-09-02 06:00:00', 
         '2009-09-02 06:30:00', 
         '2009-09-02 07:00:00', 
         '2009-09-02 08:00:00',]

times2 =   ( (2019, 9, 2, 5, 0, 0),
             (2019, 9, 2, 6, 0, 0),
             (2019, 9, 2, 6, 30, 0),
             (2019, 9, 2, 7, 0, 0),
             (2019, 9, 2, 8, 0, 0))


times3 = [ '05:00', '06:00', '06:30', '07:00', '08:00' ]

import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory
from datetime import datetime
import numpy as np

from CARR_Scenarios1_info import info as info 

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.5,8.0] # [17.0,10.0] #[12.8, 12.0]
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12 #18
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Create grid of subplots
fign, axn = plt.subplots(4, len( biotfiles ) )
fige, axe = plt.subplots(4, len( biotfiles ) )
figd, axd = plt.subplots(4, len( biotfiles ) )

# Internal routine used in loop below
def setlims(ax, col):
    # Set axis limits to max range of range of column
    
    blim = ax[0,col].get_xlim()
    ilim = ax[1,col].get_xlim()
    dlim = ax[2,col].get_xlim()
    olim = ax[3,col].get_xlim()
    lim = [ min(blim[0], ilim[0], dlim[0], olim[0]), max(blim[1], ilim[1], dlim[1], olim[1]) ]
    
    ax[0,col].set_xlim(lim)
    ax[1,col].set_xlim(lim)
    ax[2,col].set_xlim(lim)
    ax[3,col].set_xlim(lim)

    # ax[row,col].grid(False)
    # ax[row,col].set_aspect(1)
    return

# Create diretory for plots
create_directory( info['dir_plots'], 'components' )

# Memory for statistics
n = len(biotfiles)
Bm = np.zeros([3,n])
Bs = np.zeros([3,n])
Im = np.zeros([3,n])
Is = np.zeros([3,n])
Om = np.zeros([3,n])
Os = np.zeros([3,n])
Dm = np.zeros([3,n])
Ds = np.zeros([3,n])

for i in range( n ):
    biot = biotfiles[i]
    divB = divBfiles[i]
    inner = innerfiles[i]
    outer = outerfiles[i]
    
    biotdf = pd.read_pickle( os.path.join( rootdir, biot ) )
    divBdf = pd.read_pickle( os.path.join( rootdir, divB ) )
    innerdf = pd.read_pickle( os.path.join( rootdir, inner ) )
    outerdf = pd.read_pickle( os.path.join( rootdir, outer ) )
    
    sumdf = pd.DataFrame()
    sumdf[r'$B_n$'] = divBdf[r'$B_n$'] + innerdf[r'$B_n$'] + outerdf[r'$B_n$'] 
    sumdf[r'$B_e$'] = divBdf[r'$B_e$'] + innerdf[r'$B_e$'] + outerdf[r'$B_e$'] 
    sumdf[r'$B_d$'] = divBdf[r'$B_d$'] + innerdf[r'$B_d$'] + outerdf[r'$B_d$']
    sumdf['Longitude'] = divBdf['Longitude']
    sumdf['Latitude']  = divBdf['Latitude']
    sumdf['Time']      = divBdf['Time']

    Bm[0,i] = biotdf[r'$B_n$'].mean()
    Bm[1,i] = biotdf[r'$B_e$'].mean()
    Bm[2,i] = biotdf[r'$B_d$'].mean()
    Bs[0,i] = biotdf[r'$B_n$'].std()
    Bs[1,i] = biotdf[r'$B_e$'].std()
    Bs[2,i] = biotdf[r'$B_d$'].std()

    Im[0,i] = innerdf[r'$B_n$'].mean()
    Im[1,i] = innerdf[r'$B_e$'].mean()
    Im[2,i] = innerdf[r'$B_d$'].mean()
    Is[0,i] = innerdf[r'$B_n$'].std()
    Is[1,i] = innerdf[r'$B_e$'].std()
    Is[2,i] = innerdf[r'$B_d$'].std()

    Om[0,i] = outerdf[r'$B_n$'].mean()
    Om[1,i] = outerdf[r'$B_e$'].mean()
    Om[2,i] = outerdf[r'$B_d$'].mean()
    Os[0,i] = outerdf[r'$B_n$'].std()
    Os[1,i] = outerdf[r'$B_e$'].std()
    Os[2,i] = outerdf[r'$B_d$'].std()

    Dm[0,i] = divBdf[r'$B_n$'].mean()
    Dm[1,i] = divBdf[r'$B_e$'].mean()
    Dm[2,i] = divBdf[r'$B_d$'].mean()
    Ds[0,i] = divBdf[r'$B_n$'].std()
    Ds[1,i] = divBdf[r'$B_e$'].std()
    Ds[2,i] = divBdf[r'$B_d$'].std()

    ############################
    # biot savart
    ############################
    
    scattern = axn[0,i].hist( biotdf[r'$B_n$'], bins=20 )
    scattere = axe[0,i].hist( biotdf[r'$B_e$'], bins=20 )
    scatterd = axd[0,i].hist( biotdf[r'$B_d$'], bins=20 )

    ############################
    # inner
    ############################
    
    scattern = axn[1,i].hist( innerdf[r'$B_n$'], bins=20 )
    scattere = axe[1,i].hist( innerdf[r'$B_e$'], bins=20 )
    scatterd = axd[1,i].hist( innerdf[r'$B_d$'], bins=20 )

    ############################
    # divB
    ############################
    
    axn[2,i].hist( divBdf[r'$B_n$'], bins=20 )
    axe[2,i].hist( divBdf[r'$B_e$'], bins=20 )
    axd[2,i].hist( divBdf[r'$B_d$'], bins=20 )
    ############################
    # outer
    ############################
    
    axn[3,i].hist( outerdf[r'$B_n$'], bins=20 )
    axe[3,i].hist( outerdf[r'$B_e$'], bins=20 )
    axd[3,i].hist( outerdf[r'$B_d$'], bins=20 )
    
    setlims(axn,i)
    setlims(axe,i)
    setlims(axd,i)
        
# Add times to each column
for axp, col in zip(axn[0], times2):
    dtime = datetime(*col) 
    time_hhmm = dtime.strftime("%H:%M")
    axp.set_title(time_hhmm)

for axp, col in zip(axe[0], times2):
    dtime = datetime(*col) 
    time_hhmm = dtime.strftime("%H:%M")
    axp.set_title(time_hhmm)

for axp, col in zip(axd[0], times2):
    dtime = datetime(*col) 
    time_hhmm = dtime.strftime("%H:%M")
    axp.set_title(time_hhmm)

# Add titles to each row
for axp, row in zip(axn[:,0], ['Biot Savart', 'Inner', r'$\nabla \cdot \mathbf{B}$', 'Outer']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axe[:,0], ['Biot Savart', 'Inner', r'$\nabla \cdot \mathbf{B}$', 'Outer']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axd[:,0], ['Biot Savart', 'Inner', r'$\nabla \cdot \mathbf{B}$', 'Outer']):
    axp.set_ylabel(row, rotation=90)

# Add titles to each column
for axp in axn[3,:] :
    axp.set_xlabel(r'$B_N$ (nT)')

for axp in axe[3,:] :
    axp.set_xlabel(r'$B_E$ (nT)')

for axp in axd[3,:] :
    axp.set_xlabel(r'$B_D$ (nT)')
    
fign.tight_layout()
fige.tight_layout()
figd.tight_layout()
     
fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.png" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.pdf" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.eps" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.jpg" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.tif" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-hist.svg" ) )
    
fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.png" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.pdf" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.eps" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.jpg" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.tif" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-hist.svg" ) )
    
figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.png" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.pdf" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.eps" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.jpg" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.tif" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-hist.svg" ) )
    
# Create latex table with statistics
for j in range(3):
    
    print( '\\begin{center}' )
    print( '\\begin{tabular}{c c c c c}' )
    print( '\\hline' )
    print( 'Time & Biot--Savart & Inner & Outer & {\divB} \\\\ [0.5ex]' )  
    print( '(Hour) & (nT) & (nT) & (nT) & (nT) \\\\ [0.5ex]' )  
    print( '\\hline\hline' )
    
    for i in range(n):
        print( f'{times3[i]} & {Bm[j,i]:.2f} ({Bs[j,i]:.2f}) & {Im[j,i]:.2f} ({Is[j,i]:.2f}) & {Om[j,i]:.2f} ({Os[j,i]:.2f}) & {Dm[j,i]:.2f} ({Ds[j,i]:.2f}) \\\\ ')
    
    print( '\\hline' )
    print( '\\end{tabular}' )
    print( '\\end{center}' )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:08:36 2024

@author: Dean Thomas
"""

# Analyze size of divB, inner surface, and outer surface integral contributions
# to Biot-Savart

rootdir = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived/heatmaps'

biotfiles = [ 'Dean_Thomas_052924_1.3df.010800.cdf.ms-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.021600.cdf.ms-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.032400.cdf.ms-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.043200.cdf.ms-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.064800.cdf.ms-heatmap-world.pkl' ]

divBfiles = [ 'Dean_Thomas_052924_1.3df.010800.cdf.divB-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.021600.cdf.divB-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.032400.cdf.divB-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.043200.cdf.divB-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.064800.cdf.divB-heatmap-world.pkl' ]

innerfiles = [ 'Dean_Thomas_052924_1.3df.010800.cdf.inner-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.021600.cdf.inner-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.032400.cdf.inner-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.043200.cdf.inner-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.064800.cdf.inner-heatmap-world.pkl' ]

outerfiles = [ 'Dean_Thomas_052924_1.3df.010800.cdf.outer-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.021600.cdf.outer-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.032400.cdf.outer-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.043200.cdf.outer-heatmap-world.pkl',
    'Dean_Thomas_052924_1.3df.064800.cdf.outer-heatmap-world.pkl' ]

times = [ '2000-01-01 01:00:00', 
         '2000-01-01 04:00:00',
         '2000-01-01 07:00:00',
         '2000-01-01 10:00:00',
         '2000-01-01 16:00:00']

times2 =    ((2000, 1, 1, 1, 0, 0),
             (2000, 1, 1, 4, 0, 0),
             (2000, 1, 1, 7, 0, 0),
             (2000, 1, 1, 10, 0, 0),
             (2000, 1, 1, 16, 0, 0)) 


import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory
from datetime import datetime

from Dean_Thomas_052924_1_info import info as info

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.5,8.5] # [17.0,10.0] #[12.8, 12.0]
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12 #18
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Create grid of subplots
fign, axn = plt.subplots(3, len( biotfiles ) )
fige, axe = plt.subplots(3, len( biotfiles ) )
figd, axd = plt.subplots(3, len( biotfiles ) )

# Internal routine used in loop below
def setlims(ax, row, col):
    # Set axis limits to max range of x or y axis
    
    ylim = ax[row,col].get_ylim()
    xlim = ax[row,col].get_xlim()
    lim = [ min(ylim[0],xlim[0]), max(ylim[1],xlim[1]) ]
    ax[row,col].set_ylim(lim)
    ax[row,col].set_xlim(lim)

    ax[row,col].grid(False)
    ax[row,col].set_aspect(1)
    return

# Which parameter is used for colormap applied to data
color = 'Latitude'
# color = 'Longitude'

# Create diretory for plots
create_directory( info['dir_plots'], 'components' )

# print( 'Time\tType\tAvg\tStd\tMax\tMin' )

for i in range( len( biotfiles ) ):
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

    # print( times[i], '\tbiot N\t', biotdf[r'$B_n$'].mean(), '\t', biotdf[r'$B_n$'].std(), '\t', biotdf[r'$B_n$'].max(), '\t', biotdf[r'$B_n$'].min() )
    # print( times[i], '\tbiot E\t', biotdf[r'$B_e$'].mean(), '\t', biotdf[r'$B_e$'].std(), '\t', biotdf[r'$B_e$'].max(), '\t', biotdf[r'$B_e$'].min() )
    # print( times[i], '\tbiot D\t', biotdf[r'$B_d$'].mean(), '\t', biotdf[r'$B_d$'].std(), '\t', biotdf[r'$B_d$'].max(), '\t', biotdf[r'$B_d$'].min() )
    
    # print( times[i], '\tinner N\t', innerdf[r'$B_n$'].mean(), '\t', innerdf[r'$B_n$'].std(), '\t', innerdf[r'$B_n$'].max(), '\t', innerdf[r'$B_n$'].min() )
    # print( times[i], '\tinner E\t', innerdf[r'$B_e$'].mean(), '\t', innerdf[r'$B_e$'].std(), '\t', innerdf[r'$B_e$'].max(), '\t', innerdf[r'$B_e$'].min() )
    # print( times[i], '\tinner D\t', innerdf[r'$B_d$'].mean(), '\t', innerdf[r'$B_d$'].std(), '\t', innerdf[r'$B_d$'].max(), '\t', innerdf[r'$B_d$'].min() )
    
    # print( times[i], '\tdivB N\t', divBdf[r'$B_n$'].mean(), '\t', divBdf[r'$B_n$'].std(), '\t', divBdf[r'$B_n$'].max(), '\t', divBdf[r'$B_n$'].min() )
    # print( times[i], '\tdivB E\t', divBdf[r'$B_e$'].mean(), '\t', divBdf[r'$B_e$'].std(), '\t', divBdf[r'$B_e$'].max(), '\t', divBdf[r'$B_e$'].min() )
    # print( times[i], '\tdivB D\t', divBdf[r'$B_d$'].mean(), '\t', divBdf[r'$B_d$'].std(), '\t', divBdf[r'$B_d$'].max(), '\t', divBdf[r'$B_d$'].min() )
    
    # print( times[i], '\touter N\t', outerdf[r'$B_n$'].mean(), '\t', outerdf[r'$B_n$'].std(), '\t', outerdf[r'$B_n$'].max(), '\t', outerdf[r'$B_n$'].min() )
    # print( times[i], '\touter E\t', outerdf[r'$B_e$'].mean(), '\t', outerdf[r'$B_e$'].std(), '\t', outerdf[r'$B_e$'].max(), '\t', outerdf[r'$B_e$'].min() )
    # print( times[i], '\touter D\t', outerdf[r'$B_d$'].mean(), '\t', outerdf[r'$B_d$'].std(), '\t', outerdf[r'$B_d$'].max(), '\t', outerdf[r'$B_d$'].min() )

    ############################
    # inner
    ############################
    
    scattern = axn[0,i].scatter( biotdf[r'$B_n$'], innerdf[r'$B_n$'], c=innerdf[color] )
    setlims(axn,0,i)

    scattere = axe[0,i].scatter( biotdf[r'$B_e$'], innerdf[r'$B_e$'], c=innerdf[color] )
    setlims(axe,0,i)
    
    scatterd = axd[0,i].scatter( biotdf[r'$B_d$'], innerdf[r'$B_d$'], c=innerdf[color] )
    setlims(axd,0,i)

    ############################
    # divB
    ############################
    
    axn[1,i].scatter( biotdf[r'$B_n$'], divBdf[r'$B_n$'], c=divBdf[color] )
    setlims(axn,1,i)
       
    axe[1,i].scatter( biotdf[r'$B_e$'], divBdf[r'$B_e$'], c=divBdf[color] )
    setlims(axe,1,i)
    
    axd[1,i].scatter( biotdf[r'$B_d$'], divBdf[r'$B_d$'], c=divBdf[color] )
    setlims(axd,1,i)
        
    ############################
    # outer
    ############################
    
    axn[2,i].scatter( biotdf[r'$B_n$'], outerdf[r'$B_n$'], c=outerdf[color] )
    setlims(axn,2,i)

    axe[2,i].scatter( biotdf[r'$B_e$'], outerdf[r'$B_e$'], c=outerdf[color] )
    setlims(axe,2,i)

    axd[2,i].scatter( biotdf[r'$B_d$'], outerdf[r'$B_d$'], c=outerdf[color] )
    setlims(axd,2,i)
        
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
for axp, row in zip(axn[:,0], ['Inner $B_N$ (nT)', r'$\nabla \cdot \mathbf{B}$  $B_N$ (nT)', 'Outer $B_N$ (nT)']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axe[:,0], ['Inner $B_E$ (nT)', r'$\nabla \cdot \mathbf{B}$  $B_E$ (nT)', 'Outer $B_E$ (nT)']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axd[:,0], ['Inner $B_D$ (nT)', r'$\nabla \cdot \mathbf{B}$  $B_D$ (nT)', 'Outer $B_D$ (nT)']):
    axp.set_ylabel(row, rotation=90)

# Add titles to each column
for axp in axn[2,:] :
    axp.set_xlabel(r'Biot-Savart $B_N$ (nT)')

for axp in axe[2,:] :
    axp.set_xlabel(r'Biot-Savart $B_E$ (nT)')

for axp in axd[2,:] :
    axp.set_xlabel(r'Biot-Savart $B_D$ (nT)')
 
fign.subplots_adjust( bottom=0.15 )
cbar_axn = fign.add_axes([0.3,0.05,0.4,0.02]) # (left, bottom, width, height)
cbarn = fign.colorbar(scattern, cax=cbar_axn, orientation='horizontal', shrink=0.4)
cbarn.set_label(color)

fige.subplots_adjust( bottom=0.15 )
cbar_axe = fige.add_axes([0.3,0.05,0.4,0.02]) # (left, bottom, width, height)
cbare = fige.colorbar(scattere, cax=cbar_axe, orientation='horizontal', shrink=0.4)
cbare.set_label(color)

figd.subplots_adjust( bottom=0.15 )
cbar_axd = figd.add_axes([0.3,0.05,0.4,0.02]) # (left, bottom, width, height)
cbard = figd.colorbar(scatterd, cax=cbar_axd, orientation='horizontal', shrink=0.4)
cbard.set_label(color)
    
fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.png" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.pdf" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.eps" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.jpg" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.tif" ) )
# fign.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-grid.svg" ) )
    
fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.png" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.pdf" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.eps" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.jpg" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.tif" ) )
# fige.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-grid.svg" ) )
    
figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.png" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.pdf" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.eps" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.jpg" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.tif" ) )
# figd.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-grid.svg" ) )
    

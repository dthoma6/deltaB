#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:08:36 2024

@author: Dean Thomas
"""

# Analyze size of divB, inner surface, and outer surface integral contributions
# to Biot-Savart

rootdir = '/Volumes/PhysicsHD/Bob_Weigel_070323_3.derived/heatmaps'

biotfiles = [ '3d__ful_4_e20000101-010000-000.out.cdf.ms-heatmap-world.pkl',
    '3d__ful_4_e20000101-040000-000.out.cdf.ms-heatmap-world.pkl',
    '3d__ful_4_e20000101-070000-000.out.cdf.ms-heatmap-world.pkl',
    '3d__ful_4_e20000101-100000-000.out.cdf.ms-heatmap-world.pkl',
    '3d__ful_4_e20000101-160000-000.out.cdf.ms-heatmap-world.pkl' ]

divBfiles = [ '3d__ful_4_e20000101-010000-000.out.cdf.divB-heatmap-world.pkl',
    '3d__ful_4_e20000101-040000-000.out.cdf.divB-heatmap-world.pkl',
    '3d__ful_4_e20000101-070000-000.out.cdf.divB-heatmap-world.pkl',
    '3d__ful_4_e20000101-100000-000.out.cdf.divB-heatmap-world.pkl',
    '3d__ful_4_e20000101-160000-000.out.cdf.divB-heatmap-world.pkl' ]

innerfiles = [ '3d__ful_4_e20000101-010000-000.out.cdf.inner-heatmap-world.pkl',
    '3d__ful_4_e20000101-040000-000.out.cdf.inner-heatmap-world.pkl',
    '3d__ful_4_e20000101-070000-000.out.cdf.inner-heatmap-world.pkl',
    '3d__ful_4_e20000101-100000-000.out.cdf.inner-heatmap-world.pkl',
    '3d__ful_4_e20000101-160000-000.out.cdf.inner-heatmap-world.pkl' ]

outerfiles = [ '3d__ful_4_e20000101-010000-000.out.cdf.outer-heatmap-world.pkl',
    '3d__ful_4_e20000101-040000-000.out.cdf.outer-heatmap-world.pkl',
    '3d__ful_4_e20000101-070000-000.out.cdf.outer-heatmap-world.pkl',
    '3d__ful_4_e20000101-100000-000.out.cdf.outer-heatmap-world.pkl',
    '3d__ful_4_e20000101-160000-000.out.cdf.outer-heatmap-world.pkl' ]

times = [ '2000-01-01 01:00:00', 
         '2000-01-01 04:00:00',
         '2000-01-01 07:00:00',
         '2000-01-01 10:00:00',
         '2000-01-01 16:00:00']

times2 =  ((2000, 1, 1, 1, 0, 0),
          (2000, 1, 1, 4, 0, 0),
          (2000, 1, 1, 7, 0, 0),
          (2000, 1, 1, 10, 0, 0),
          (2000, 1, 1, 16, 0, 0)) 

import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory
from datetime import datetime

from Bob_Weigel_070323_3_info import info as info

# Set some plot configs
plt.rcParams["figure.figsize"] = [14.0,8.0] #[12.5,8.0] # [17.0,10.0] #[12.8, 12.0]
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12 #18
plt.rcParams.update({
    # "text.usetex": True,
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
    import math
    
    ylim = ax[row,col].get_ylim()
    xlim = ax[row,col].get_xlim()
    # lim = [ min(ylim[0],xlim[0]), max(ylim[1],xlim[1]) ]
    
    bot = min(ylim[0],xlim[0])
    top = max(ylim[1],xlim[1])
    tb = max( abs(bot), top )
    if( tb > 10 ):
        tb2 = int(math.ceil(tb / 10.0)) * 10 
    else:
        tb2 = int(math.ceil(tb))
    lim = [-tb2,tb2]
    
    ax[row,col].set_ylim(lim)
    ax[row,col].set_xlim(lim)

    ax[row,col].set_yticks([int(lim[0]-0.5),0,int(lim[1]+0.5)])
    ax[row,col].set_xticks([int(lim[0]-0.5),0,int(lim[1]+0.5)])

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
    
    ######################################################################################
    ######################################################################################
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap region
    ######################################################################################
    ######################################################################################

    sumdf = pd.DataFrame()
    sumdf[r'$B_n$'] = - divBdf[r'$B_n$'] + innerdf[r'$B_n$'] - outerdf[r'$B_n$'] 
    sumdf[r'$B_e$'] = - divBdf[r'$B_e$'] + innerdf[r'$B_e$'] - outerdf[r'$B_e$'] 
    sumdf[r'$B_d$'] = - divBdf[r'$B_d$'] + innerdf[r'$B_d$'] - outerdf[r'$B_d$']
    sumdf['Longitude'] = divBdf['Longitude']
    sumdf['Latitude']  = divBdf['Latitude']
    sumdf['Time']      = divBdf['Time']

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
    
    ######################################################################################
    ######################################################################################
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap region
    ######################################################################################
    ######################################################################################

    axn[1,i].scatter( biotdf[r'$B_n$'], - divBdf[r'$B_n$'], c=divBdf[color] )
    setlims(axn,1,i)
       
    axe[1,i].scatter( biotdf[r'$B_e$'], - divBdf[r'$B_e$'], c=divBdf[color] )
    setlims(axe,1,i)
    
    axd[1,i].scatter( biotdf[r'$B_d$'], - divBdf[r'$B_d$'], c=divBdf[color] )
    setlims(axd,1,i)
        
    ############################
    # outer
    ############################
    
    ######################################################################################
    ######################################################################################
    # Verify that Biot-Savert = - outer surf integral - inner surf integral - divB integral
    # Note, that my inner integral is already negative, I calculate the outer
    # integral for the gap region
    ######################################################################################
    ######################################################################################

    axn[2,i].scatter( biotdf[r'$B_n$'], - outerdf[r'$B_n$'], c=outerdf[color] )
    setlims(axn,2,i)

    axe[2,i].scatter( biotdf[r'$B_e$'], - outerdf[r'$B_e$'], c=outerdf[color] )
    setlims(axe,2,i)

    axd[2,i].scatter( biotdf[r'$B_d$'], - outerdf[r'$B_d$'], c=outerdf[color] )
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
for axp, row in zip(axn[:,0], [r'$\mathsf{B_{in}}$ (nT)', 
                               r'$\mathsf{B_{div}}$ (nT)', 
                               r'$\mathsf{B_{out}}$ (nT)']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axe[:,0], [r'$\mathsf{B_{in}}$ (nT)', 
                               r'$\mathsf{B_{div}}$ (nT)', 
                               r'$\mathsf{B_{out}}$ (nT)']):
    axp.set_ylabel(row, rotation=90)

for axp, row in zip(axd[:,0], [r'$\mathsf{B_{in}}$ (nT)', 
                               r'$\mathsf{B_{div}}$ (nT)', 
                               r'$\mathsf{B_{out}}$ (nT)']):
    axp.set_ylabel(row, rotation=90)

# Add titles to each column
for axp in axn[2,:] :
    axp.set_xlabel(r'$\mathsf{B_{BS}}$ (nT)')

for axp in axe[2,:] :
    axp.set_xlabel(r'$\mathsf{B_{BS}}$ (nT)')

for axp in axd[2,:] :
    axp.set_xlabel(r'$\mathsf{B_{BS}}$ (nT)')
 
# Set title
fign.suptitle(r'$\mathsf{B_N}$ for each term')
fige.suptitle(r'$\mathsf{B_E}$ for each term')
figd.suptitle(r'$\mathsf{B_D}$ for each term')

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
    

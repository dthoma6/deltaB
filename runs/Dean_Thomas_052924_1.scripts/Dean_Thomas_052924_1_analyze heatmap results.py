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

import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory

from Dean_Thomas_052924_1_info import info as info

# True if we plot results, False if we only print statistics
PLOT = True

print( 'Time\tType\tAvg\tStd\tMax\tMin' )

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

    print( times[i], '\tbiot N\t', biotdf[r'$B_n$'].mean(), '\t', biotdf[r'$B_n$'].std(), '\t', biotdf[r'$B_n$'].max(), '\t', biotdf[r'$B_n$'].min() )
    print( times[i], '\tbiot E\t', biotdf[r'$B_e$'].mean(), '\t', biotdf[r'$B_e$'].std(), '\t', biotdf[r'$B_e$'].max(), '\t', biotdf[r'$B_e$'].min() )
    print( times[i], '\tbiot D\t', biotdf[r'$B_d$'].mean(), '\t', biotdf[r'$B_d$'].std(), '\t', biotdf[r'$B_d$'].max(), '\t', biotdf[r'$B_d$'].min() )
    
    print( times[i], '\tinner N\t', innerdf[r'$B_n$'].mean(), '\t', innerdf[r'$B_n$'].std(), '\t', innerdf[r'$B_n$'].max(), '\t', innerdf[r'$B_n$'].min() )
    print( times[i], '\tinner E\t', innerdf[r'$B_e$'].mean(), '\t', innerdf[r'$B_e$'].std(), '\t', innerdf[r'$B_e$'].max(), '\t', innerdf[r'$B_e$'].min() )
    print( times[i], '\tinner D\t', innerdf[r'$B_d$'].mean(), '\t', innerdf[r'$B_d$'].std(), '\t', innerdf[r'$B_d$'].max(), '\t', innerdf[r'$B_d$'].min() )
    
    print( times[i], '\tdivB N\t', divBdf[r'$B_n$'].mean(), '\t', divBdf[r'$B_n$'].std(), '\t', divBdf[r'$B_n$'].max(), '\t', divBdf[r'$B_n$'].min() )
    print( times[i], '\tdivB E\t', divBdf[r'$B_e$'].mean(), '\t', divBdf[r'$B_e$'].std(), '\t', divBdf[r'$B_e$'].max(), '\t', divBdf[r'$B_e$'].min() )
    print( times[i], '\tdivB D\t', divBdf[r'$B_d$'].mean(), '\t', divBdf[r'$B_d$'].std(), '\t', divBdf[r'$B_d$'].max(), '\t', divBdf[r'$B_d$'].min() )
    
    print( times[i], '\touter N\t', outerdf[r'$B_n$'].mean(), '\t', outerdf[r'$B_n$'].std(), '\t', outerdf[r'$B_n$'].max(), '\t', outerdf[r'$B_n$'].min() )
    print( times[i], '\touter E\t', outerdf[r'$B_e$'].mean(), '\t', outerdf[r'$B_e$'].std(), '\t', outerdf[r'$B_e$'].max(), '\t', outerdf[r'$B_e$'].min() )
    print( times[i], '\touter D\t', outerdf[r'$B_d$'].mean(), '\t', outerdf[r'$B_d$'].std(), '\t', outerdf[r'$B_d$'].max(), '\t', outerdf[r'$B_d$'].min() )

    if PLOT:
        color = 'Latitude'
        levels = 6
        
        create_directory( info['dir_plots'], 'components' )
    
        ############################
        # divB
        ############################
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_n$'], divBdf[r'$B_n$'], c=divBdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="upper right", title=color)
        ax.add_artist(legend)
        plt.xlabel(r'Biot-Savart $B_N$ (nT)' )
        plt.ylabel(r'$\nabla \cdot B$  $B_N$ (nT)')
        plt.title('BATSRUS '  + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color +'-Bn-' + times[i] + "-divB-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-divB-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-divB-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-divB-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-divB-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-divB-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_e$'], divBdf[r'$B_e$'], c=divBdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="upper right", title=color)
        plt.xlabel(r'Biot-Savart $B_E$ (nT)')
        plt.ylabel(r'$\nabla \cdot B$  $B_E$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
        
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-divB-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-divB-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-divB-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-divB-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-divB-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-divB-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_d$'], divBdf[r'$B_d$'], c=divBdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="upper right", title=color)
        plt.xlabel(r'Biot-Savart $B_D$ (nT)')
        plt.ylabel(r'$\nabla \cdot B$  $B_D$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-divB-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-divB-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-divB-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-divB-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-divB-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-divB-biot.svg" ) )
        
        ############################
        # inner
        ############################
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_n$'], innerdf[r'$B_n$'], c=innerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        ax.add_artist(legend)
        plt.xlabel(r'Biot-Savart $B_N$ (nT)' )
        plt.ylabel(r'Inner $B_N$ (nT)')
        plt.title('BATSRUS '  + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-inner-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-inner-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-inner-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-inner-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-inner-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-inner-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_e$'], innerdf[r'$B_e$'], c=innerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_E$ (nT)')
        plt.ylabel(r'Inner $B_E$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
        
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-inner-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-inner-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-inner-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-inner-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-inner-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-inner-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_d$'], innerdf[r'$B_d$'], c=innerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_D$ (nT)')
        plt.ylabel(r'Inner $B_D$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-inner-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-inner-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-inner-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-inner-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-inner-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-inner-biot.svg" ) )
        
        ############################
        # outer
        ############################
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_n$'], outerdf[r'$B_n$'], c=outerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        ax.add_artist(legend)
        plt.xlabel(r'Biot-Savart $B_N$ (nT)' )
        plt.ylabel(r'Outer $B_N$ (nT)')
        plt.title('BATSRUS '  + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-outer-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-outer-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-outer-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-outer-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-outer-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-outer-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_e$'], outerdf[r'$B_e$'], c=outerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_E$ (nT)')
        plt.ylabel(r'Outer $B_E$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
        
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-outer-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-outer-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-outer-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-outer-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-outer-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-outer-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_d$'], outerdf[r'$B_d$'], c=outerdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_D$ (nT)')
        plt.ylabel(r'Outer $B_D$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-outer-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-outer-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-outer-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-outer-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-outer-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-outer-biot.svg" ) )
        
        ############################
        # sum divB + inner + outer
        ############################
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_n$'], sumdf[r'$B_n$'], c=sumdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        ax.add_artist(legend)
        plt.xlabel(r'Biot-Savart $B_N$ (nT)' )
        plt.ylabel(r'Sum $B_N$ (nT)')
        plt.title('BATSRUS '  + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bn-' + times[i] + "-sum-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-sum-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-sum-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-sum-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-sum-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bn-' + times[i] + "-sum-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_e$'], sumdf[r'$B_e$'], c=sumdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_E$ (nT)')
        plt.ylabel(r'Sum $B_E$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
        
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Be-' + times[i] + "-sum-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-sum-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-sum-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-sum-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-sum-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Be-' + times[i] + "-sum-biot.svg" ) )
        
        fig, ax = plt.subplots()
        scatter = ax.scatter( biotdf[r'$B_d$'], sumdf[r'$B_d$'], c=sumdf[color] )
        legend = ax.legend(*scatter.legend_elements(num=levels), loc="lower right", title=color)
        plt.xlabel(r'Biot-Savart $B_D$ (nT)')
        plt.ylabel(r'Sum $B_D$ (nT)')
        plt.title('BATSRUS ' + times[i] )
        plt.show()
    
        fig.savefig( os.path.join( info['dir_plots'], 'components', color + '-Bd-' + times[i] + "-sum-biot.png" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-sum-biot.pdf" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-sum-biot.eps" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-sum-biot.jpg" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-sum-biot.tif" ) )
        # fig.savefig( os.path.join( info['dir_plots'], 'components', 'Bd-' + times[i] + "-sum-biot.svg" ) )
    
 
    
    

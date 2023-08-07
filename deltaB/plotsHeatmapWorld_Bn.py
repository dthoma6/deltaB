#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:45:41 2023

@author: Dean Thomas
"""

import pandas as pd
import logging

from spacepy import coordinates as coord
from spacepy.time import Ticktock
import os.path
import seaborn as sns
import matplotlib.pyplot as plt
# from matplotlib.colors import SymLogNorm
from datetime import datetime
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade


from deltaB import calc_ms_b_paraperp, calc_ms_b_region,\
    calc_iono_b, calc_gap_b, \
    convert_BATSRUS_to_dataframe, \
    date_timeISO, create_directory

# Example of info = {...}
# data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

# info = {
#         "model": "SWMF",
#         "run_name": "DIPTSUR2",
#         "rCurrents": 4.0,
#         "rIonosphere": 1.01725,
#         "file_type": "out",
#         "dir_run": os.path.join(data_dir, "DIPTSUR2"),
#         "dir_plots": os.path.join(data_dir, "DIPTSUR2.plots"),
#         "dir_derived": os.path.join(data_dir, "DIPTSUR2.derived"),
# }

def loop_heatmapworld_ms(info, times, nlat, nlong):
    """Loop thru data in BATSRUS files to create data for heat maps showing the 
    breakdown of Bn due to currents parallel and perpendicular to B field.  
    Results will be used to generate heatmaps of Bn from these currents over 
    surface of earth.

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
    Outputs:
        None - other than the pickle file that is generated
    """

    # We will walk around the globe collecting B field estimates,
    # the spacing of lat and long samples
    dlat = 180. / nlat
    dlong = 360. / nlong

    n = nlat * nlong
    
    # Storage for results
    Bn = [None] * n
    Bparan = [None] * n
    Bperpn = [None] * n
    Bperpphin = [None] * n
    Bperpphiresn = [None] * n
    B_lat = [None] * n
    B_long = [None] * n
    B_time = [None] * n

    # Loop through the files
    for p in range(len(times)):
        time = times[p]

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
    
        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong}')
    
                # k is counter to keep track of where we're at in storing the results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with BATSRUS data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xgsm = Xgeo.convert('GSM', 'car')
                X = Xgsm.data[0]
    
                # Get the B field at the point X and ISO time using the BATSRUS data
                # results are in SM coordinates
                Bn[k], Bparan[k], Bperpn[k], Bperpphin[k], Bperpphiresn[k], = \
                    calc_ms_b_paraperp(X, timeISO, df)
        
        # Determine the fraction of the B field due to various currents - those
        # parallell to the B field, perpendicular to B field and in the phi-hat
        # direction (jperpenddicular dot phi-hat), and the remaining perpendicular 
        # current (jperpendicular - jperpendicular dot phi-hat).
        B_fraction_parallel = [m/n for m, n in zip(Bparan, Bn)]
        B_fraction_perp = [m/n for m, n in zip(Bperpn, Bn)]
        B_fraction_perpphi = [m/n for m, n in zip(Bperpphin, Bn)]
        B_fraction_perpphires = [m/n for m, n in zip(Bperpphiresn, Bn)]
    
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total': Bn, 
                            r'Parallel': Bparan, 
                            r'Perpendicular': Bperpn, 
                            r'Perpendicular $\phi$': Bperpphin, 
                            r'Perpendicular Residual': Bperpphiresn,
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time,
                            r'Fraction Parallel': B_fraction_parallel, 
                            r'Fraction Perpendicular': B_fraction_perp, 
                            r'Fraction Perpendicular $\phi$': B_fraction_perpphi, 
                            r'Fraction Perpendicular Residual': B_fraction_perpphires } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.ms-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def loop_heatmapworld_ms_by_region(info, times, nlat, nlong, deltamp, deltabs, thicknessns, nearradius, 
                       mpfiles, bsfiles, nsfiles):
    """Loop thru data in BATSRUS files to create data for heat maps showing the 
    breakdown of Bn due to currents in the different regions.  
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth
    Results will be used to generate heatmaps of Bn from these currents over 
    surface of earth.

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
                    
        mpfiles, bsfiles, nsfiles = list of filenames (located in os.path.join( info['dir_derived'], 'mp-bs-ns') 
                that contain the magnetopause, bow shock, and neutral sheet locations 
                at time time
    Outputs:
        None - other than the pickle file that is generated
    """

    # We will walk around the globe collecting B field estimates,
    # the spacing of lat and long samples
    dlat = 180. / nlat
    dlong = 360. / nlong

    n = nlat * nlong
    
    # Storage for results
    Bn = [None] * n
    Bnother = [None] * n
    Bnmag = [None] * n
    Bnneu = [None] * n
    Bnnear = [None] * n
    B_lat = [None] * n
    B_long = [None] * n
    B_time = [None] * n

    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Loop through the files
    for p in range(len(times)):
        time = times[p]
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
 
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong}')
    
                # k is counter to keep track of where we're at in storing the results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with BATSRUS data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xgsm = Xgeo.convert('GSM', 'car')
                X = Xgsm.data[0]
    
                # Get the B field at the point X and ISO time using the BATSRUS data
                # results are in SM coordinates
                Bn[k], Bnother[k], Bnmag[k], Bnneu[k], Bnnear[k], = \
                   calc_ms_b_region( info, deltamp, deltabs, thicknessns, nearradius, 
                                          time, mpfiles[p], bsfiles[p], nsfiles[p], X )
        
        # Determine the fraction of the B field due to various currents - those
        # parallell to the B field, perpendicular to B field and in the phi-hat
        # direction (jperpenddicular dot phi-hat), and the remaining perpendicular 
        # current (jperpendicular - jperpendicular dot phi-hat).
        B_fraction_other = [m/n for m, n in zip(Bnother, Bn)]
        B_fraction_mag = [m/n for m, n in zip(Bnmag, Bn)]
        B_fraction_neu = [m/n for m, n in zip(Bnneu, Bn)]
        B_fraction_near = [m/n for m, n in zip(Bnnear, Bn)]
    
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total': Bn, 
                            r'Other': Bnother, 
                            r'Magnetosheath': Bnmag, 
                            r'Neutral Sheet': Bnneu, 
                            r'Near Earth': Bnnear,
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time,
                            r'Fraction Other': B_fraction_other, 
                            r'Fraction Magnetosheath': B_fraction_mag, 
                            r'Fraction Neutral Sheet': B_fraction_neu, 
                            r'Fraction Near Earth': B_fraction_near } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmapworld_ms( info, times, vmin, vmax, nlat, nlong ):
    """Plot results from loop_heatmap_ms, showing the heatmap of
    Bn contributions from currents parallel and perpendicular to the local
    B field over the surface of the earth

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples
        
    Outputs:
        None - other than the plot files saved
    """

    # To use different nlat/nlong values the code below must be generalized.
    # It assumes nlat = 9 and nlong = 12.  See, for example, the calls to
    # set the xticks and yticks below    
    # assert( nlat == 9 )
    # assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Calculate the B field.  The results will be saved in a pickle file
        # loop_heatmap(info, base, nlat, nlong)
                        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_heatmap_ms above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where Colaba is
        colabalatlong = [18.907, 72.815]

        # Create the heatmaps
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        # Draw (1,0,0) SM point on graph.  Note, that the x and y axes are the 
        # the column and row numbers of the heatmap.  So prorate the lat long
        # accordingly. x axis is 0 to nlong, y axis is 0 to nlat

        # Draw (1,0,0) SM point on graph.  Note, that the x and y axes are the 
        # the column and row numbers of the heatmap.  So prorate the lat long
        # accordingly. x axis is 0 to nlong, y axis is 0 to nlat

        colabaxy = [(6 + (colabalatlong[1])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Longitude', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'ms-total-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', 'Parallel' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\parallel}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-parallel-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular $\phi$' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp \cdot \hat \phi}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-phi-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular Residual' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-residue-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
    return

def plot_heatmapworld_ms_by_region( info, times, vmin, vmax, nlat, nlong, deltamp, deltabs, 
                       thicknessns, nearradius ):
    """Plot results from loop_heatmap_ms, showing the heatmap of
    Bn contributions from currents in the regions of the BATSRUS grid.
    
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples

        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
                           
    Outputs:
        None - other than the plot files saved
    """

    # To use different nlat/nlong values the code below must be generalized.
    # It assumes nlat = 9 and nlong = 12.  See, for example, the calls to
    # set the xticks and yticks below    
    # assert( nlat == 9 )
    # assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']
    
    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Calculate the B field.  The results will be saved in a pickle file
        # loop_heatmap(info, base, nlat, nlong)
                        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_heatmap_ms above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where Colaba is
        colabalatlong = [18.907, 72.815]

        # Create the heatmaps
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        # Draw (1,0,0) SM point on graph.  Note, that the x and y axes are the 
        # the column and row numbers of the heatmap.  So prorate the lat long
        # accordingly. x axis is 0 to nlong, y axis is 0 to nlat

        # Draw (1,0,0) SM point on graph.  Note, that the x and y axes are the 
        # the column and row numbers of the heatmap.  So prorate the lat long
        # accordingly. x axis is 0 to nlong, y axis is 0 to nlat

        colabaxy = [(6 + (colabalatlong[1])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        
        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Longitude', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'ms-all-regions-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', 'Other' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{other}}}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-other-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Magnetosheath' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{magnetosheath}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-magnetosheath-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Neutral Sheet' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{neutral-sheet}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-neutral-sheet-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Near Earth' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{near-earth}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-near-earth-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
    return

def loop_heatmapworld_iono(info, times, nlat, nlong):
    """Loop thru data in RIM files to create data for heat maps showing the 
    breakdown of Bn due to Pedersen and Hall currents in the ionosphere.  Results 
    will be used to generate heatmaps of Bn over surface of earth

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
    Outputs:
        None - other than the pickle file that is generated
    """

    # We will walk around the globe collecting B field estimates,
    # the spacing of lat and long samples
    dlat = 180. / nlat
    dlong = 360. / nlong

    n = nlat * nlong
    
    # Storage for results
    Bnp = [None] * n
    Bep = [None] * n
    Bdp = [None] * n
    Bxp = [None] * n
    Byp = [None] * n
    Bzp = [None] * n
    Bnh = [None] * n
    Beh = [None] * n
    Bdh = [None] * n
    Bxh = [None] * n
    Byh = [None] * n
    Bzh = [None] * n
    B_lat = [None] * n
    B_long = [None] * n
    B_time = [None] * n

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
            
        # We need the filepath for BATSRUS file
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
    
        # # Read in the BATSRUS file 
        # df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong}')
    
                # k is counter to keep track of where we're at in storing the results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into SM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xsm = Xgeo.convert('SM', 'car')
                X = Xsm.data[0]
    
                # Get the B field at the point X and timeiso using the RIM data
                # results are in SM coordinates
                Bnp[k], Bep[k], Bdp[k], Bxp[k], Byp[k], Bzp[k], Bnh[k], Beh[k], Bdh[k], Bxh[k], Byh[k], Bzh[k] = \
                    calc_iono_b(X, filepath, timeISO, info['rCurrents'], info['rIonosphere'])
    
        # Determine the fraction of the B field due to various currents - those
        # parallell to the B field, perpendicular to B field and in the phi-hat
        # direction (jperpenddicular dot phi-hat), and the remaining perpendicular 
        # current (jperpendicular - jperpendicular dot phi-hat).
        # b_fraction = [m/n for m, n in zip(b_original_parallel, b_original)]
        # b_fraction_perp = [m/n for m, n in zip(b_original_perp, b_original)]
        # b_fraction_perpphi = [m/n for m, n in zip(b_original_perpphi, b_original)]
        # b_fraction_perpphires = [m/n for m, n in zip(b_original_perpphires, b_original)]
    
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total Pedersen': Bnp, 
                            r'Total Hall': Bnh,
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.iono-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmapworld_iono( info, times, vmin, vmax, nlat, nlong ):
    """Plot results from loop_heatmap_iono, showing the heatmap of
    Bn contributions from Pedersend and Hall currents in the inonosphere

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        
    Outputs:
        None - other than the plot files saved
    """
    
    # To use different nlat/nlong values the code below must be generalized.
    # It assumes nlat = 9 and nlong = 12.  See, for example, the calls to
    # set the xticks and yticks below    
    # assert( nlat == 9 )
    # assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_sum_db above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.iono-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        

        # Determine where Colaba is
        colabalatlong = [18.907, 72.815]

        # Create the heatmaps
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        colabaxy = [(6 + (colabalatlong[1])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Longitude', values='Total Pedersen' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to $\mathbf{j}_{Pedersen}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'iono-pedersen-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Longitude', values='Total Hall' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to $\mathbf{j}_{Hall}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'iono-hall-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )

def loop_heatmapworld_gap(info, times, nlat, nlong):
    """Loop thru data in RIM files to create plots showing the breakdown of
    Bn due to field aligned currents in the gap region.  Results 
    will be used to generate heatmaps of Bn over surface of earth

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
    Outputs:
        None - other than the pickle file that is generated
    """

    # We will walk around the globe collecting B field estimates,
    # the spacing of lat and long samples
    dlat = 180. / nlat
    dlong = 360. / nlong

    n = nlat * nlong
    
    # Storage for results
    Bn = [None] * n
    Be = [None] * n
    Bd = [None] * n
    Bx = [None] * n
    By = [None] * n
    Bz = [None] * n
    B_lat = [None] * n
    B_long = [None] * n
    B_time = [None] * n

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
            
        # We need the filepath for BATSRUS file
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
    
        # # Read in the BATSRUS file 
        # df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong}')
    
                # k is counter to keep track of where we're at in storing the results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xsm = Xgeo.convert('SM', 'car')
                X = Xsm.data[0]
    
                # Get the B field at the point X and timeiso using the RIM data
                # results are in SM coordinates
                Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] = \
                    calc_gap_b(X, filepath, timeISO, info['rCurrents'], info['rIonosphere'], 30, 30, 30)
    
        # Determine the fraction of the B field due to various currents - those
        # parallell to the B field, perpendicular to B field and in the phi-hat
        # direction (jperpenddicular dot phi-hat), and the remaining perpendicular 
        # current (jperpendicular - jperpendicular dot phi-hat).
        # b_fraction = [m/n for m, n in zip(b_original_parallel, b_original)]
        # b_fraction_perp = [m/n for m, n in zip(b_original_perp, b_original)]
        # b_fraction_perpphi = [m/n for m, n in zip(b_original_perpphi, b_original)]
        # b_fraction_perpphires = [m/n for m, n in zip(b_original_perpphires, b_original)]
    
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total': Bn, 
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.gap-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmapworld_gap( info, times, vmin, vmax, nlat, nlong ):
    """Plot results from loop_heatmap_gap, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        
    Outputs:
        None - other than the plot files saved
    """
    
    # To use different nlat/nlong values the code below must be generalized.
    # It assumes nlat = 9 and nlong = 12.  See, for example, the calls to
    # set the xticks and yticks below    
    # assert( nlat == 9 )
    # assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_sum_db above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where Colaba is
        colabalatlong = [18.907, 72.815]

        # Create the heatmaps
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        # Draw (1,0,0) SM point on graph.  Note, that the x and y axes are the 
        # the column and row numbers of the heatmap.  So prorate the lat long
        # accordingly. x axis is 0 to nlong, y axis is 0 to nlat

        colabaxy = [(6 + (colabalatlong[1])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Longitude', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to Gap $\mathbf{j}_\parallel$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'gap-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
     
    return

def earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax, title, params):   
    """Plot results from loop_heatmap_gap, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
        time = UTC time of plot
        
        nlong, nlat = number of longitude and latitude bins

        ax = subplot where plot will be placed
        
        title = title for plot
        
    Outputs:
        None - other than the plot generated
        
    """
    # Time for the data in the file (hour:minutes)
    time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)

    # We need the filepath for BATSRUS file to get pickle file
    filepath = info['files']['magnetosphere'][time]
    basename = os.path.basename(filepath)

    pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
    pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname)
    
    df = pd.read_pickle(pklpath)

    # Draw map with day/night
    ax.coastlines()
    dtime = datetime(*time) 
    ax.add_feature(Nightshade(dtime, alpha=0.1))   
    
    # Get lat/longs for heatmap
    lon_bins = np.array(df['Longitude'])
    lat_bins = np.array(df['Latitude'])
    density_bins = np.array(df[title])
    
    lon_bins_2d = lon_bins.reshape(nlat,nlong)
    lat_bins_2d = lat_bins.reshape(nlat,nlong)
    density = density_bins.reshape(nlat,nlong)
    
    # Determine where Colaba is
    colabalatlong = [18.907, 72.815]
    ax.plot(colabalatlong[1], colabalatlong[0], markersize=5, color='yellow', marker='*', zorder=6, alpha=0.8)
    
    # Colormap for heatmap
    cmap = plt.colormaps['coolwarm']
    
    # Draw heatmap
    im = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Set ticks
    ax.set_xticks([-90,0,90], crs=ccrs.PlateCarree())
    ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
    
    # Draw colorbar and title
    plt.colorbar(mappable=im, ax=ax, orientation='vertical', shrink=0.4, fraction=0.1, pad=0.02)
    # ax.set_title(r'B\textsubscript{N} (nT) from ' + title + ' (' + time_hhmm +')' )
    return im

def plot_heatmapworld_ms_by_region_grid(info, times, vmin, vmax, nlat, nlong, deltamp, deltabs, thicknessns, nearradius):
    """Plot results from loop_heatmap_by_region, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
       info = info on files to be processed, see info = {...} example above
            
       times = the times associated with the files for which we will create
           heatmaps
        
        nlat, nlong = number of longitude and latitude bins

        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
                    
    Outputs:
        None - other than the plot generated
        
    """

    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [12.8, 10.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    
    cols = len(times)
    
    fig, ax = plt.subplots(4,cols, sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(right=0.5)
    
    for i in range(cols):
        time = times[i]
        
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[0,i], 'Magnetosheath',  params)
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[1,i], 'Near Earth', params)
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[2,i], 'Neutral Sheet', params)
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[3,i], 'Other', params)
        ax[3,i].set_xlabel('Longitude')
    
    ax[0,0].set_ylabel('Latitude')
    ax[1,0].set_ylabel('Latitude')
    ax[2,0].set_ylabel('Latitude')
    ax[3,0].set_ylabel('Latitude')
  
    for axp, col in zip(ax[0], times):
        time_hhmm = str(col[3]).zfill(2) + ':' + str(col[4]).zfill(2)
        axp.set_title(r'B\textsubscript{N} (nT) ' + time_hhmm)

    for axp, row in zip(ax[:,0], ['Magnetosheath\nLatitude', 'Near Earth\nLatitude', 'Neutral Sheet\nLatitude', 'Other\nLatitude']):
        axp.set_ylabel(row, rotation=90)
   
    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.png" ) )
    return

def earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax, title, pklpath):   
    """Plot results from loop_heatmap_gap, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
        time = UTC time of plot
        
        nlong, nlat = number of longitude and latitude bins

        ax = subplot where plot will be placed
        
        title = title for plot
        
    Outputs:
        None - other than the plot generated
        
    """
    # Time for the data in the file (hour:minutes)
    time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)

    # Get data
    df = pd.read_pickle(pklpath)

    # Draw map with day/night
    ax.coastlines()
    dtime = datetime(*time) 
    ax.add_feature(Nightshade(dtime, alpha=0.1))   

    # Get lat/longs for heatmap
    lon_bins = np.array(df['Longitude'])
    lat_bins = np.array(df['Latitude'])
    
    if title == 'MS $j_{\perp Residual}$':
        density_bins = np.array(df[r'Perpendicular Residual'])
    if title == 'MS $j_{\perp \phi}$':
        density_bins = np.array(df[r'Perpendicular $\phi$'])
    if title == 'MS $j_\parallel$':
        density_bins = np.array(df[r'Parallel' ])
    if title == 'Gap $j_\parallel$':
        density_bins = np.array(df[r'Total' ])
    if title == '$j_{Pederson}$':
        density_bins = np.array(df[r'Total Pedersen' ])
    if title == '$j_{Hall}$':
        density_bins = np.array(df[r'Total Hall' ])
    
    if title == 'Fraction MS $j_{\perp Residual}$':
        density_bins = np.array(df[r'Fraction Perpendicular Residual'])
    if title == 'Fraction MS $j_{\perp \phi}$':
        density_bins = np.array(df[r'Fraction Perpendicular $\phi$'])
    if title == 'Fraction MS $j_\parallel$':
        density_bins = np.array(df[r'Fraction Parallel' ])
    if title == 'Fraction Gap $j_\parallel$':
        density_bins = np.array(df[r'Fraction Total' ])
    if title == 'Fraction $j_{Pederson}$':
        density_bins = np.array(df[r'Fraction Total Pedersen' ])
    if title == 'Fraction $j_{Hall}$':
        density_bins = np.array(df[r'Fraction Total Hall' ])
    
    lon_bins_2d = lon_bins.reshape(nlat,nlong)
    lat_bins_2d = lat_bins.reshape(nlat,nlong)
    density = density_bins.reshape(nlat,nlong)
    
    # Determine where Colaba is
    colabalatlong = [18.907, 72.815]
    ax.plot(colabalatlong[1], colabalatlong[0], markersize=5, color='yellow', marker='*', zorder=6, alpha=0.8)

    # Colormap for heatmap
    cmap = plt.colormaps['coolwarm']
    
    # Add colormap
    im = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_xticks([-90,0,90], crs=ccrs.PlateCarree())
    ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
    
    # Add colorbar and titles
    plt.colorbar(mappable=im, ax=ax, orientation='vertical', shrink=0.4, fraction=0.1, pad=0.02)
    # ax.set_title(r'$B_N$ (nT) from ' + title + ' (' + time_hhmm +')' )
    return 

def plot_heatmapworld_ms_by_currents_grid(info, times, vmin, vmax, nlat, nlong):
    """Plot results from loop_heatmap_by_region, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
       info = info on files to be processed, see info = {...} example above
            
       times = the times associated with the files for which we will create
           heatmaps
        
        nlat, nlong = number of longitude and latitude bins

        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
                    
    Outputs:
        None - other than the plot generated
        
    """

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [12.8, 12.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    cols = len(times)
    
    fig, ax = plt.subplots(6,cols, sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(right=0.5)
    
    for i in range(cols):
        time = times[i]
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[0,i], 'MS $j_\parallel$', pklpath)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[1,i], 'MS $j_{\perp \phi}$',  pklpath)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[2,i], 'MS $j_{\perp Residual}$',  pklpath)
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[3,i], 'Gap $j_\parallel$', pklpath)
        
        pklname = basename + '.iono-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
       
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[4,i], '$j_{Pederson}$', pklpath)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[5,i], '$j_{Hall}$', pklpath)
        ax[5,i].set_xlabel('Longitude')
    
    ax[0,0].set_ylabel('Latitude')
    ax[1,0].set_ylabel('Latitude')
    ax[2,0].set_ylabel('Latitude')
    ax[3,0].set_ylabel('Latitude')
    ax[4,0].set_ylabel('Latitude')
    ax[5,0].set_ylabel('Latitude')
  
    for axp, col in zip(ax[0], times):
        time_hhmm = str(col[3]).zfill(2) + ':' + str(col[4]).zfill(2)
        axp.set_title(r'B\textsubscript{N} (nT) ' + time_hhmm)

    for axp, row in zip(ax[:,0], ['Magnetosphere $j_{\parallel}$\nLatitude', 'Magnetosphere $j_{\perp \phi}$\nLatitude', \
                                  'Magnetosphere $\Delta j_{\perp}$\nLatitude', 'Gap $j_{\parallel}$\nLatitude', \
                                  'Ionosphere $j_{P}$\nLatitude', 'Ionosphere $j_{H}$\nLatitude']):
        axp.set_ylabel(row, rotation=90)

    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.png" ) )
    return


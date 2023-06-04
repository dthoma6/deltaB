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

from deltaB import calc_ms_b_paraperp, \
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

def loop_heatmap_ms(info, times, nlat, nlong):
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
    for i in range(len(times)):
        time = times[i]

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
    
        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]
        
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
                long = noonlatlong[2] + 180. - (j + 0.5)*dlong
                if( long > 180 ): long = long - 360
                if( long < -180 ): long = long + 360
                B_long[k] = long
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with BATSRUS data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph')
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
        pklname = basename + '.ms-heatmap.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmap_ms( info, times, vmin, vmax, nlat, nlong ):
    """Plot results from loop_heatmap_ms, showing the heatmap of
    Bn contributions from currents parallel and perpendicular to the local
    B field over the surface of the earth

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
    assert( nlat == 9 )
    assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Calculate the B field.  The results will be saved in a pickle file
        # loop_heatmap(info, base, nlat, nlong)
                        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4])
        
        # Read the pickle file with the data from loop_heatmap_ms above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]
        
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

        noonxy = [6, nlat - (noonlatlong[1] + 90)*nlat/180]
        colabaxy = [(6 + (colabalatlong[1]-noonlatlong[2])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Time', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'ms-total-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', 'Parallel' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\parallel}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-parallel-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular $\phi$' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp \cdot \hat \phi}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-phi-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular Residual' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-residue-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
    return

def loop_heatmap_iono(info, times, nlat, nlong):
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
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]
        
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
                long = noonlatlong[2] + 180. - (j + 0.5)*dlong
                if( long > 180 ): long = long - 360
                if( long < -180 ): long = long + 360
                B_long[k] = long
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into SM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph')
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
        pklname = basename + '.iono-heatmap.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmap_iono( info, times, vmin, vmax, nlat, nlong ):
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
    assert( nlat == 9 )
    assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4])
        
        # Read the pickle file with the data from loop_sum_db above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.iono-heatmap.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]

        # Determine where Colaba is
        colabalatlong = [18.907, 72.815]

        # Create the heatmaps
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        noonxy = [6, nlat - (noonlatlong[1] + 90)*nlat/180]
        colabaxy = [(6 + (colabalatlong[1]-noonlatlong[2])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Time', values='Total Pedersen' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to $\mathbf{j}_{Pedersen}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'iono-pedersen-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Time', values='Total Hall' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to $\mathbf{j}_{Hall}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'iono-hall-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )

def loop_heatmap_gap(info, times, nlat, nlong):
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
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]
        
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
                long = noonlatlong[2] + 180. - (j + 0.5)*dlong
                if( long > 180 ): long = long - 360
                if( long < -180 ): long = long + 360
                B_long[k] = long
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph')
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
        pklname = basename + '.gap-heatmap.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    return

def plot_heatmap_gap( info, times, vmin, vmax, nlat, nlong ):
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
    assert( nlat == 9 )
    assert( nlong == 12 )
    
    # Colorscale for heatmap
    cmap = plt.colormaps['coolwarm']

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4])
        
        # Read the pickle file with the data from loop_sum_db above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
        
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )
        
        # Determine where noon is, that is longitude for (1,0,0) SM
        noonsm = coord.Coords([[1,0,0]], 'SM', 'car')
        noonsm.ticks = Ticktock([timeISO], 'ISO')
        noongeo = noonsm.convert('GEO', 'sph')
        noonlatlong = noongeo.data[0]

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

        noonxy = [6, nlat - (noonlatlong[1] + 90)*nlat/180]
        colabaxy = [(6 + (colabalatlong[1]-noonlatlong[2])*nlong/360)%nlong, \
                    nlat - (colabalatlong[0] + 90)*nlat/180]

        fig = plt.gcf()
        df1 = df.pivot(index='Latitude', columns='Time', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", annot_kws={"size":6})
        plt.scatter( noonxy[0], noonxy[1], marker='o', color='black' )
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Local Time')
        ax.set_ylabel('GEO Latitude')
        ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to Gap $\mathbf{j}_\parallel$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'gap-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
     
    return
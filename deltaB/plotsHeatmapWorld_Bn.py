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
from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
import swmfio

from deltaB import find_regions, calc_ms_b_paraperp, calc_ms_b_region,\
    calc_iono_b, calc_gap_b, calc_gap_b_rim, \
    convert_BATSRUS_to_dataframe, \
    date_timeISO, create_directory

# Colormap used in heatmaps below
COLORMAP = 'coolwarm'

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

def plot_heatmapworld( info, time, vmin, vmax, nlat, nlong, pklname, pltstr, 
                      pltname, title, csys='GEO', threesixty=False ):
    """Generic routine for creating heatmaps.  Called by other routines below
    to generate heatmaps for magnetosphere, gap region, and ionosphere.
    
    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        time = the time associated with the file for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples
        
        pklname = path for pickle file containing data to be plotted
        
        pltstr = name of column in pklfile used to create heatmap
        
        pltname = filename for storing plot
        
        title = plot title

        csys = coordinate system for plots (e.g., GEO, SM)
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
    Outputs:
        None - other than the plot files saved
    """
    
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    from itertools import repeat
    
    # Read the pickle file with the data 
    # Pickle file in info['dir_derived'] folder
    df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
    
    # If needed transform from 'GEO' to csys
    if csys != 'GEO':
        # Get the ISO time from the filepath
        timeISO = date_timeISO( time )

        # Create list of radii, lats and longs
        lat = df['Latitude']
        lon = df['Longitude']
        r = np.ones( len(lat) )
        latlon1 = list( zip( r, lat, lon ) )
        
        # Use spacepy for coordinate transformation
        tim = list( repeat( timeISO, len(lat) ) )
        X1 = coord.Coords(latlon1, 'GEO', 'sph', use_irbem=False)
        X1.ticks = Ticktock(tim, 'ISO')
        X2 = X1.convert( csys, 'sph' )
        latlon2 = X2.data
        
        # Extract lats and longs
        lat3 = []
        lon3 = []
        for i in range(len(latlon2)):
            lat3.append( latlon2[i][1] )
            lon3.append( latlon2[i][2] )
    else:
        # Data in pickle file is in 'GEO' if cys == 'GEO'
        lat3 = df['Latitude']
        lon3 = df['Longitude']

    lat3 = np.array(lat3)
    lon3 = np.array(lon3)
    total = np.array(df[pltstr])

    # if vmin or vmax are not specified, base them on limits of data
    # Vmin/vmax set range of heatmap colormap
    if vmin is None or vmax is None:
        vmin1 = np.min( total )
        vmax1 = np.max( total )
    else:
        vmin1 = vmin
        vmax1 = vmax

    # Reshape bins to 2D meshes
    lon_bins_2d = lon3.reshape(nlat,nlong)
    lat_bins_2d = lat3.reshape(nlat,nlong)
    density = total.reshape(nlat,nlong)
    
    # Colormap for heatmap
    cmap = plt.colormaps[COLORMAP]
    
    # Is the map 0->360 or -180->180 in longitude
    if threesixty:
        proj = ccrs.PlateCarree(central_longitude=180.)
    else:
        proj = ccrs.PlateCarree()

    # Create heatmap
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, subplot_kw={'projection': proj})
    im = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density, cmap=cmap, vmin=vmin1, 
                       vmax=vmax1, transform=ccrs.PlateCarree())
    
    # Set ticks and axis labels
    if threesixty:
        ax.set_xticks([90,180,270], crs=ccrs.PlateCarree())
    else:
        ax.set_xticks([-90,0,90], crs=ccrs.PlateCarree())
        
    ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
    
    lon_formatter = LongitudeFormatter(direction_label=True)
    lat_formatter = LatitudeFormatter(direction_label=True)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
    
    ax.set_xlabel(csys + ' Longitude')
    ax.set_ylabel(csys + ' Latitude')

    # Draw colorbar
    plt.colorbar(mappable=im, ax=ax, orientation='vertical', shrink=0.4, 
                 fraction=0.1, pad=0.02)

    # Save plot
    plt.title(title)
    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
     
    return

def loop_heatmapworld_ms(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in BATSRUS files to create data for heat maps showing the 
    breakdown of Bn due to currents parallel and perpendicular to B field.  
    Results will be used to generate heatmaps of Bn from these currents over 
    surface of earth.

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file that is saved
    """

    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_ms( p, times, deltahr ):
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

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB heatmap for... {basename}')

        # Read in the BATSRUS file 
        df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
        
        # Get the ISO time
        if deltahr is None:
            timeISO = date_timeISO( times[p] )
        else:
            dtime = datetime(*times[p]) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()

        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each point
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong} for {basename}')
    
                # k is counter to keep track of where to store results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember, we must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with BATSRUS data.  Our point is on the earth's surface, so the
                # first entry (radius) is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xgsm = Xgeo.convert('GSM', 'car')
                X = Xgsm.data[0]
    
                # Get the B field at the point X and ISO time using the BATSRUS data
                # results are in SM coordinates
                Bn[k], Bparan[k], Bperpn[k], Bperpphin[k], Bperpphiresn[k] = \
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

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        Parallel(n_jobs=num_cores)(delayed(wrap_ms)(p, times, deltahr) for p in range(len(times)))
    else:
        for p in range(len(times)):
            wrap_ms(p, times, deltahr)

    return

def loop_heatmapworld_ms_by_region(info, times, nlat, nlong, deltamp, deltabs, 
                                    thicknessns, nearradius, mpfiles, bsfiles, 
                                    nsfiles, deltahr=None, maxcores = 20):
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
                    
        mpfiles, bsfiles, nsfiles = list of filenames 
                (located in os.path.join( info['dir_derived'], 'mp-bs-ns') 
                that contain the magnetopause, bow shock, and neutral sheet locations 
                at time time
        
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
                
    Outputs:
        None - other than the pickle file that is saved
    """

    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_by_region( p, times, deltahr, deltamp, deltabs, thicknessns, \
                       nearradius, mpfiles, bsfiles, nsfiles, params ):
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

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
 
        logging.info(f'Calculate magnetosphere dB by region heatmap for... {basename}')

        # Get the ISO time
        if deltahr is None:
            timeISO = date_timeISO( times[p] )
        else:
            dtime = datetime(*times[p]) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        logging.info(f'Parsing BATSRUS file... {os.path.basename(filepath)}')
        batsrus = swmfio.read_batsrus(filepath)

        # Identify region for each point in BATSRUS grid
        other, magnetosheath, neutralsheet, nearearth, region, bowshock, \
            magnetopause = find_regions( info, batsrus, deltamp, deltabs, 
                       thicknessns, nearradius, 
                       times[p], mpfiles[p], bsfiles[p], nsfiles[p],   
                       interpType = 'nearest')
            
        logging.info('Calculating delta B contributions...')

        # Convert BATSRUS data to dataframes for each region
        df = convert_BATSRUS_to_dataframe(batsrus, info['rCurrents'], region=region)
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong} for {basename}')
    
                # k is counter to keep track of where we store the results
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
                Bn[k], Bnother[k], Bnmag[k], Bnneu[k], Bnnear[k] = \
                    calc_ms_b_region( X, times[p], df )
        
        # Determine the fraction of the B field due to various regions in the
        # magnetosphere
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

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )

    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'        
    
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        Parallel(n_jobs=num_cores)(delayed(wrap_by_region)(p, times, deltahr,\
                                    deltamp, deltabs, \
                                    thicknessns, nearradius, \
                                    mpfiles, bsfiles, \
                                    nsfiles, params) for p in range(len(times)))
    else:
        for p in range(len(times)):
            wrap_by_region(p, times, deltahr,\
                        deltamp, deltabs, \
                        thicknessns, nearradius, \
                        mpfiles, bsfiles, nsfiles, params)
    return

def plot_heatmapworld_ms_total( info, times, vmin, vmax, nlat, nlong, csys='GEO', 
                               threesixty=False ):
    """Plot results from loop_heatmapworld_ms, showing the heatmap of total Bn 
    contributions from all magnetosphere contributions.

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples

        csys = coordinate system for plots (e.g., GEO, SM)
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
    Outputs:
        None - other than the plot files saved
    """
    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Read the pickle file with the data from loop_heatmapworld_ms above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        pltstr = 'Total'

        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Create names
        title = r'$B_N$ (nT) due to MS $\mathbf{j}$ ' + f'({time_hhmm})'
        pltname = 'ms-heatmap-' +  csys + '-' + str(time[0]) + str(time[1]) + \
            str(time[2]) + '-' + str(time[3]) + '-' + str(time[4]) + '.png'
            
        # Make plot
        plot_heatmapworld( info, time, vmin, vmax, nlat, nlong, pklname, pltstr, 
                          pltname, title, csys=csys, threesixty=threesixty )
    return

def plot_heatmapworld_ms( info, times, vmin, vmax, nlat, nlong ):
    """Plot results from loop_heatmapworld_ms, showing the heatmap of
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
    cmap = plt.colormaps[COLORMAP]

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_heatmap_ms above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
                
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
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}$ ' + f'({time_hhmm})')
        plt.show()
        pltname = 'ms-total-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        create_directory( info['dir_plots'], 'heatmaps' )
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', 'Parallel' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{\parallel}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-parallel-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular $\phi$' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{\perp \cdot \hat \phi}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-phi-heatmap-' +  str(time[0]) + str(time[1]) + \
            str(time[2]) + '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Longitude', r'Perpendicular Residual' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-perpendicular-residue-heatmap-' +  str(time[0]) + str(time[1]) + \
            str(time[2]) + '-' + str(time[3]) + '-' + str(time[4]) + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
       
    return

def plot_heatmapworld_ms_by_region( info, times, vmin, vmax, nlat, nlong, deltamp, deltabs, 
                       thicknessns, nearradius ):
    """Plot results from loop_heatmapworld_ms_by_region, showing the heatmap of
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
    
    # Colorscale for heatmap
    cmap = plt.colormaps[COLORMAP]
    
    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Loop through the files
    for i in range(len(times)):
        time = times[i]
                                
        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Read the pickle file with the data from loop_heatmapworld_ms_by_region above
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
        df = pd.read_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )
                
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
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
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
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
        plt.scatter( colabaxy[0], colabaxy[1], marker='+', color='black')
        # ax.set_xticks([0,3,6,9,12])
        # ax.set_xticklabels(['24:00','06:00','12:00','18:00','24:00'])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('GEO Latitude')
        # ax.set_yticklabels(['80','60','40','20','0', '-20', '-40', '-60', '-80'])
        plt.title(r'$B_N$ (nT) due to MS $\mathbf{j}_{other}$' + f' ({time_hhmm})')
        plt.show()
        pltname = 'ms-other-heatmap-' +  str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.' + params + '.png'
        fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', pltname ) )
        
        fig = plt.gcf()
        df1 = df.pivot('Latitude', 'Time', r'Magnetosheath' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
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
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
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
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax, annot=True, fmt=".0f", 
                         annot_kws={"size":6})
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

def loop_heatmapworld_iono(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in RIM files to create data for heat maps showing the 
    breakdown of Bn due to Pedersen and Hall currents in the ionosphere.  Results 
    will be used to generate heatmaps of Bn over surface of earth

    Inputs:
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        None - other than the pickle file that is generated
    """

    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_iono( p, times, deltahr ):
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
        
        # We need the filepath for RIM file
        # We only search for the nearest minute, ignoring last entry in key
        time = times[p]
        for key in info['files']['ionosphere']:
            if( key[0] == time[0] and key[1] == time[1] and key[2] == time[2] and \
                key[3] == time[3] and key[4] == time[4] ):
                    filepath = info['files']['ionosphere'][key]
                    
        # filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        
        logging.info(f'Calculate ionosphere dB heatmap for... {basename}')

        # Get the ISO time
        if deltahr is None:
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
        
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong} for {basename}')
        
                # k is counter to keep track of where we store the results
                k = i*nlong + j
        
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into SM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry (radius) is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xsm = Xgeo.convert('SM', 'car')
                X = Xsm.data[0]
        
                # Get the B field at the point X and timeiso using the RIM data
                # results are in SM coordinates
                Bnp[k], Bep[k], Bdp[k], Bxp[k], Byp[k], Bzp[k], Bnh[k], Beh[k], Bdh[k], \
                    Bxh[k], Byh[k], Bzh[k] = \
                    calc_iono_b(X, filepath, timeISO, info['rCurrents'], info['rIonosphere'])
        
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total Pedersen': Bnp, 
                            r'Total Hall': Bnh,
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.iono-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        Parallel(n_jobs=num_cores)(delayed(wrap_iono)(p, times, deltahr) \
                                   for p in range(len(times)))
    else:
        for p in range(len(times)):
            wrap_iono(p, times, deltahr)

    return

def plot_heatmapworld_iono( info, times, vmin, vmax, nlat, nlong, csys='GEO', 
                           threesixty=False ):
    """Plot results from loop_heatmapworld_iono, showing the heatmap of
    Bn contributions from ionospheric currents (Pedersen and Hall).

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples

        csys = coordinate system for plots (e.g., GEO, SM)
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
    Outputs:
        None - other than the plot files saved
    """
    
    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Read the pickle file with the data from loop_heatmapworld_iono above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.iono-heatmap-world.pkl'

        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Create names
        pltstr = 'Total Pedersen'
        title = r'$B_N$ (nT) due to $\mathbf{j}_{Pedersen}$ ' + f'({time_hhmm})'
        pltname = 'iono-pedersen-heatmap-' +  csys + '-' + str(time[0]) + str(time[1]) + \
            str(time[2]) + '-' + str(time[3]) + '-' + str(time[4]) + '.png'
            
        # Make plot
        plot_heatmapworld( info, time, vmin, vmax, nlat, nlong, pklname, pltstr, 
                          pltname, title, csys=csys, threesixty=threesixty )

        # Create names
        pltstr = 'Total Hall'
        title = r'$B_N$ (nT) due to $\mathbf{j}_{Hall}$ ' + f'({time_hhmm})'
        pltname = 'iono-hall-heatmap-' +  csys + '-' + str(time[0]) + str(time[1]) + \
            str(time[2]) + '-' + str(time[3]) + '-' + str(time[4]) + '.png'
            
        # Make plot
        plot_heatmapworld( info, time, vmin, vmax, nlat, nlong, pklname, pltstr, 
                          pltname, title, csys=csys, threesixty=threesixty )

    return

def loop_heatmapworld_gap(info, times, nlat, nlong, nTheta=30, nPhi=30, nR=30,
                          deltahr=None, maxcores=20, useRIM=False):
    """Loop thru data in RIM files to create plots showing the breakdown of
    Bn due to field aligned currents in the gap region.  Results 
    will be used to generate heatmaps of Bn over surface of earth

    Inputs:
        
        info = locations of key directories and other info on data 
        
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]

        nlat, nlong = number of latitude and longitude samples
                    
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents
            
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

        maxcores = for parallel processing, the maximum number of cores to use
        
        useRIM = Boolean, if False use calc_gap_b, if True use calc_gap_b_rim
            The difference is calc_gap_b makes no assumptions about the RIM file
            while calc_gap_b assumes a structure to the RIM file based on 
            reverse engineering.
        
    Outputs:
        None - other than the pickle file that is generated
    """

    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_gap( p, times, deltahr, nTheta, nPhi, nR ):
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
            
        # We need the filepath for RIM file
        # We only search for the nearest minute, ignoring last entry in key
        time = times[p]
        for key in info['files']['ionosphere']:
            if( key[0] == time[0] and key[1] == time[1] and key[2] == time[2] and \
                key[3] == time[3] and key[4] == time[4] ):
                    filepath = info['files']['ionosphere'][key]

        # filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
            
        logging.info(f'Calculate gap dB heatmap for... {basename}')

        # Get the ISO time
        if deltahr is None:
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
        
        # Loop through the lat and long points on the earth's surface.
        # We will determine the B field at each pont
        for i in range(nlat):
            for j in range(nlong):
    
                logging.info(f'======== Examining {i} of {nlat}, {j} of {nlong} for {basename}')
    
                # k is counter to keep track of where we're at in storing the results
                k = i*nlong + j
    
                # Store the lat and long, which is at the center of each cell
                # Remember,we  must have -180 < longitude < +180            
                B_lat[k] = 90. - (i + 0.5)*dlat
                B_long[k] = 180. - (j + 0.5)*dlong
                B_time[k] = (j + 0.5) * 24. / nlong
                
                # We need to convert the lat-long into GSM coordiantes for use
                # with RIM data.  Our point is on the earth's surface, so the
                # first entry (radius) is 1.
                Xlatlong=[1., B_lat[i*nlong + j], B_long[i*nlong + j]]
                Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph', use_irbem=False)
                Xgeo.ticks = Ticktock([timeISO], 'ISO')
                Xsm = Xgeo.convert('SM', 'car')
                X = Xsm.data[0]
    
                # Get the B field at the point X and timeiso using the RIM data
                # results are in SM coordinates
                if useRIM:
                    Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] = \
                        calc_gap_b_rim(X, filepath, timeISO, info['rCurrents'], 
                                   info['rIonosphere'], nR)
                else:
                    Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] = \
                        calc_gap_b(X, filepath, timeISO, info['rCurrents'], 
                                   info['rIonosphere'], nTheta, nPhi, nR)
    
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'Total': Bn, 
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.gap-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        Parallel(n_jobs=num_cores)(delayed(wrap_gap)(p, times, deltahr, nTheta, nPhi, nR) 
                                   for p in range(len(times)))
    else:
        for p in range(len(times)):
            wrap_gap(p, times, deltahr, nTheta, nPhi, nR)

    return

def plot_heatmapworld_gap( info, times, vmin, vmax, nlat, nlong, csys='GEO', 
                          threesixty=False ):
    """Plot results from loop_heatmapworld_gap, showing the heatmap of
    Bn contributions from field aligned currents in the gap region.

    Inputs:
        info = info on files to be processed, see info = {...} example above
             
        times = the times associated with the files for which we will create
            heatmaps

        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples

        csys = coordinate system for plots (e.g., GEO, SM)
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
    Outputs:
        None - other than the plot files saved
    """
    # Loop through the files
    for i in range(len(times)):
        time = times[i]
        
        # Read the pickle file with the data from loop_heatmapworld_gap above
        filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        pltstr = 'Total'

        # Time for the data in the file (hour:minutes)
        time_hhmm = str(time[3]) + ':' + str(time[4]).zfill(2)
        
        # Create names
        title = r'$B_N$ (nT) due to Gap $\mathbf{j}_\parallel$ ' + f'({time_hhmm})'
        pltname = 'gap-heatmap-' +  csys + '-' + str(time[0]) + str(time[1]) + str(time[2]) + \
            '-' + str(time[3]) + '-' + str(time[4]) + '.png'
            
        # Make plot
        plot_heatmapworld( info, time, vmin, vmax, nlat, nlong, pklname, pltstr, 
                          pltname, title, csys=csys, threesixty=threesixty )
    return

def earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax, title, params, 
                         threesixty, axisticks, deltahr):   
    """Plot results from loop_heatmapworld_ms_by_region, showing the heatmap of
    Bn contributions a specific magnetospheric region at a specific time

    Inputs:
        info = info on files to be processed, see info = {...} example above

        time = UTC time of plot
        
        vmin, vmax = min/max limits of heatmap color scale
        
        nlat, nlong = number of latitude and longitude samples

        ax = subplot where plot will be placed
        
        title = title for plot, also specifies which region is plotted.  That is,
            data is in df[title] stored in pickle file
        
        params = string with run information, used in filenames, etc. 
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
        axisticks = Boolean, are x and y axis ticks included
        
       deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        im = pseudocolor color plot
           - the plot generated
        
    """
    # We need the filepath for BATSRUS file to get pickle file
    filepath = info['files']['magnetosphere'][time]
    basename = os.path.basename(filepath)

    pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
    pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname)
    
    df = pd.read_pickle(pklpath)

    # Draw map with day/night
    ax.coastlines()
    if deltahr is None:
        dtime = datetime(*time)
    else:
        dtime = datetime(*time) + timedelta(hours=deltahr)
    ax.add_feature(Nightshade(dtime, alpha=0.1))   
    
    # Get lat/longs for heatmap
    lon_bins = np.array(df['Longitude'])
    lat_bins = np.array(df['Latitude'])
    density_bins = np.array(df[title])
    
    # if vmin or vmax are not specified, base them on limits of data
    if vmin is None or vmax is None:
        vmin = np.min( density_bins )
        vmax = np.max( density_bins )
    
    # Reshape bins to 2D meshes
    lon_bins_2d = lon_bins.reshape(nlat,nlong)
    lat_bins_2d = lat_bins.reshape(nlat,nlong)
    density = density_bins.reshape(nlat,nlong)
    
    # Determine where Colaba is
    colabalatlong = [18.907, 72.815]
    ax.plot(colabalatlong[1], colabalatlong[0], markersize=5, color='yellow', 
            marker='*', zorder=6, alpha=0.8, transform=ccrs.PlateCarree())
    
    # Colormap for heatmap
    cmap = plt.colormaps[COLORMAP]
    
    # Draw heatmap
    im = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density, cmap=cmap, vmin=vmin, 
                       vmax=vmax, transform=ccrs.PlateCarree())
    
    # Set ticks
    if axisticks:
        if threesixty:
            ax.set_xticks([90,180,270], crs=ccrs.PlateCarree())
        else:
            ax.set_xticks([-90,0,90], crs=ccrs.PlateCarree())
            
        ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
        
        lon_formatter = LongitudeFormatter(direction_label=True)
        lat_formatter = LatitudeFormatter(direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
    else:
        ax.set_yticks([], crs=ccrs.PlateCarree()) # Used only if we don't display ticks
        ax.set_xticks([], crs=ccrs.PlateCarree()) # Used only if we don't display ticks
    
    # Draw colorbar and title
    # plt.colorbar(mappable=im, ax=ax, orientation='vertical', shrink=0.4, fraction=0.1, pad=0.02)
    return im

def plot_heatmapworld_ms_by_region_grid(info, times, vmin, vmax, nlat, nlong, deltamp, 
                                        deltabs, thicknessns, nearradius, 
                                        threesixty=False, axisticks=False,
                                        deltahr=None):
    """Plot heatmaps in a grid, showing Bn contributions from each magnetospheric
    region.

    Inputs:
        info = info on files to be processed, see info = {...} example above
            
        times = the times associated with the files for which we will create
           heatmaps
        
        vmin, vmax = min/max limits of heatmap color scale
    
        nlat, nlong = number of longitude and latitude bins
    
        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
            
        threesixty = Boolean, is our map 0->360 or -180->180 in longitude
        
        axisticks = Boolean, do we include x and y axis ticks on heatmaps
                    
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the plot generated
        
    """

    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [8.5,5.25] # [17.0,10.0] #[12.8, 12.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12 #18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    
    # One column of plots for each time
    cols = len(times)

    # Is the map 0->360 or -180->180 in longitude
    if threesixty:
        proj = ccrs.PlateCarree(central_longitude=180.)
    else:
        proj = ccrs.PlateCarree()

    # Create grid of subplots, 4 regions and 1 column for each time in times
    fig, ax = plt.subplots(5,cols, sharex=True, sharey=True, subplot_kw={'projection': proj})
    
    # Create heatmaps
    for i in range(cols):
        time = times[i]
        
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[0,i], 
                             'Magnetosheath',  params, threesixty, axisticks,
                             deltahr)
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[1,i], 
                             'Near Earth', params, threesixty, axisticks,
                             deltahr)
        earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[2,i],
                             'Neutral Sheet', params, threesixty, axisticks,
                             deltahr)
        im = earth_region_heatmap( info, time, vmin, vmax, nlat, nlong, ax[3,i], 
                                  'Other', params, threesixty, axisticks,
                                  deltahr)
        # ax[3,i].set_xlabel('Longitude')
    
    # Add titles to each column
    for axp, col in zip(ax[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(time_hhmm)

    # Add titles to each row identifying region
    for axp, row in zip(ax[:,0], ['Magneto-\nsheath', 'Near\nEarth', 'Neutral\nSheet', 'Other']):
        axp.set_ylabel(row, rotation=90)
   
    # Add colorbar
    cbar = fig.colorbar( im, ax=ax[4,:], orientation='horizontal' )
    cbar.set_label(r'$B_{N}$ (nT)')
    for colp in range(cols): 
        fig.delaxes(ax=ax[4,colp])

    # Set title
    fig.suptitle(r'$B_{N}$ due to Geospace Regions')

    # Save plot
    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.png" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.pdf" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.eps" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.jpg" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.tif" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-region-grid.svg" ) )
    return

def earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax, title, pklpath, 
                           threesixty, axisticks, deltahr):   
    """Plot results from loop_heatmapworld_ms, showing the heatmap of
    Bn contributions from various currents in the magnetosphere.

    Inputs:
        time = UTC time of plot
        
        vmin, vmax = min/max limits of heatmap color scale

        nlat, nlong = number of latitude and longitude samples

        ax = subplot where plot will be placed
        
        title = title for plot, also specifies which current system is plotted
        
        pklpath = path to pickle file
        
        threesixty = Boolean, is map 0->360 or -180->180 longitude
        
        axisticks = Boolean, are x and y ticks placed on heatmaps
        
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        im - pseudocolor
           - the plot generated
        
    """
    # Get data
    df = pd.read_pickle(pklpath)

    # Draw map with day/night
    ax.coastlines()
    if deltahr is None:
        dtime = datetime(*time)
    else:
        dtime = datetime(*time) + timedelta(hours=deltahr)
    ax.add_feature(Nightshade(dtime, alpha=0.1))   

    # Get lat/longs for heatmap
    lon_bins = np.array(df['Longitude'])
    lat_bins = np.array(df['Latitude'])
    
    # Select current system to be plotted
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
    
    # if vmin or vmax are not specified, base them on limits of data
    if vmin is None or vmax is None:
        vmin = np.min( density_bins )
        vmax = np.max( density_bins )
    
    lon_bins_2d = lon_bins.reshape(nlat,nlong)
    lat_bins_2d = lat_bins.reshape(nlat,nlong)
    density = density_bins.reshape(nlat,nlong)
    
    # Determine where Colaba is
    colabalatlong = [18.907, 72.815]
    ax.plot(colabalatlong[1], colabalatlong[0], markersize=5, color='yellow', 
            marker='*', zorder=6, alpha=0.8, transform=ccrs.PlateCarree())

    # Colormap for heatmap
    cmap = plt.colormaps[COLORMAP]
    
    # Add colormap
    im = ax.pcolormesh(lon_bins_2d, lat_bins_2d, density, cmap=cmap, vmin=vmin, 
                       vmax=vmax, transform=ccrs.PlateCarree())
    
    # Set ticks, changes based on whether longitude is 0->360 or -180->180
    if axisticks:
        if threesixty:
            ax.set_xticks([90,180,270], crs=ccrs.PlateCarree())
        else:
            ax.set_xticks([-90,0,90], crs=ccrs.PlateCarree())
            
        ax.set_yticks([-45,0,45], crs=ccrs.PlateCarree())
        
        lon_formatter = LongitudeFormatter(direction_label=True)
        lat_formatter = LatitudeFormatter(direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
    else:
        ax.set_yticks([], crs=ccrs.PlateCarree()) # Used only if we don't display ticks
        ax.set_xticks([], crs=ccrs.PlateCarree()) # Used only if we don't display ticks

    # Add colorbar
    # plt.colorbar(mappable=im, ax=ax, orientation='vertical', shrink=0.2, fraction=0.1, pad=0.02)
    # plt.colorbar(mappable=im, ax=ax, orientation='horizontal', shrink=0.2, fraction=0.1, pad=0.02)
    return im

def plot_heatmapworld_ms_by_currents_grid(info, times, vmin, vmax, nlat, nlong, 
                                          threesixty = False, axisticks = False,
                                          deltahr=None):
    """Plot results from loop_heatmapworld_ms, showing the heatmap of
    Bn contributions from magnetospheric currents.

    Inputs:
       info = info on files to be processed, see info = {...} example above
            
       times = the times associated with the files for which we will create
           heatmaps
        
       vmin, vmax = min/max limits of heatmap color scale
       
       nlat, nlong = number of longitude and latitude bins
       
       threesixty = Boolean, is our map 0->360 or -180->180 in longitude
       
       axisticks = Boolean, do we include x and y axis ticks on heatmaps
                    
       deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the plot generated
        
    """

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [8.5, 7.0] #[17.0,14.0] #[12.8, 12.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12 #18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    cols = len(times)
    
    # Is our map 0->360 or -180->180
    if threesixty:
        proj = ccrs.PlateCarree(central_longitude=180.)
    else:
        proj = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(7, cols, sharex=True, sharey=True, subplot_kw={'projection': proj})
     
    for i in range(cols):
        time = times[i]
        
        # We need the filepath for BATSRUS file to find the pickle filename
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        # Create heatmaps for different currents
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[0,i], 
                               'MS $j_\parallel$', pklpath, threesixty, axisticks,
                               deltahr)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[1,i], 
                               'MS $j_{\perp \phi}$',  pklpath, threesixty, axisticks,
                               deltahr)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[2,i], 
                               'MS $j_{\perp Residual}$',  pklpath, threesixty, 
                               axisticks, deltahr)
        
        # We need the filepath for RIM file to find the pickle filename
        # We only search for the nearest minute, ignoring last entry in key
        for key in info['files']['ionosphere']:
            if( key[0] == time[0] and key[1] == time[1] and key[2] == time[2] and \
                key[3] == time[3] and key[4] == time[4] ):
                    filepath = info['files']['ionosphere'][key]
                    
        # filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        # Create heatmaps
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[3,i], 
                               'Gap $j_\parallel$', pklpath, threesixty, axisticks,
                               deltahr)
        
        # Rinse and repeat for ionosphere
        pklname = basename + '.iono-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
       
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[4,i], 
                               '$j_{Pederson}$', pklpath, threesixty, axisticks,
                               deltahr)
        im = earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax[5,i], 
                                    '$j_{Hall}$', pklpath, threesixty, axisticks,
                                    deltahr)
        # ax[5,i].set_xlabel('Longitude')
              
    # Set titles for each column
    for axp, col in zip(ax[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(time_hhmm)

    # Set titles for each row
    # for axp, row in zip(ax[:,0], ['Magnetosphere $j_{\parallel}$', \
    #                               'Magnetosphere $j_{\perp \phi}$', \
    #                               'Magnetosphere $\Delta j_{\perp}$', \
    #                               'Gap $j_{\parallel}$', \
    #                               'Ionosphere $j_{P}$', \
    #                               'Ionosphere $j_{H}$']):
    #     axp.set_ylabel(row, rotation=90)
    for axp, row in zip(ax[:,0], ['$j_{\parallel}$', \
                                  '$j_{\perp \phi}$', \
                                  '$\Delta j_{\perp}$', \
                                  'Gap $j_{\parallel}$', \
                                  '$j_{P}$', \
                                  '$j_{H}$']):
        axp.set_ylabel(row, rotation=90)

    # Add colorbar
    cbar = fig.colorbar( im, ax=ax[6,:], orientation='horizontal' )
    cbar.set_label(r'$B_{N}$ (nT)')
    for colp in range(cols): 
        fig.delaxes(ax=ax[6,colp])

    # Set title
    fig.suptitle(r'$B_{N}$ due to Magnetospheric, Gap, and Ionospheric Currents')

    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.png" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.pdf" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.eps" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.jpg" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.tif" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-currents-grid.svg" ) )
    return

def plot_heatmapworld_ms_by_currents_grid2(info, times, vmin, vmax, nlat, nlong, 
                                          threesixty = False, axisticks = False,
                                          deltahr=None):
    """Plot results from loop_heatmapworld_ms, showing the heatmap of
    Bn contributions from magnetospheric currents. Produces the same plots as
    plot_heatmapworld_ms_by_currents_grid except the graphs are spilt into 2 
    files

    Inputs:
       info = info on files to be processed, see info = {...} example above
            
       times = the times associated with the files for which we will create
           heatmaps
        
       vmin, vmax = min/max limits of heatmap color scale
       
       nlat, nlong = number of longitude and latitude bins
       
       threesixty = Boolean, is our map 0->360 or -180->180 in longitude
       
       axisticks = Boolean, do we include x and y axis ticks on heatmaps
                    
       deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the plot generated
        
    """

    # Set some plot configs
    # plt.rcParams["figure.figsize"] = [17.0,8.0] 
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12 #18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    cols = len(times)
    
    # Is our map 0->360 or -180->180
    if threesixty:
        proj = ccrs.PlateCarree(central_longitude=180.)
    else:
        proj = ccrs.PlateCarree()
    
    # Create fig1 for magnetospheric currents and fig2 for gap & ionospheric currents
    plt.rcParams["figure.figsize"] = [8.5,3.75] #[17.0,6.1] 
    fig1, ax1 = plt.subplots(3, cols, sharex=True, sharey=True, subplot_kw={'projection': proj})
    plt.rcParams["figure.figsize"] = [8.5,4.5] #[17.0,8.0] 
    fig2, ax2 = plt.subplots(4, cols, sharex=True, sharey=True, subplot_kw={'projection': proj})
    
    for i in range(cols):
        time = times[i]
        
        # We need the filepath for BATSRUS file to find the pickle filename
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        # Create heatmaps for different currents
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax1[0,i], 
                               'MS $j_\parallel$', pklpath, threesixty, axisticks,
                               deltahr)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax1[1,i], 
                               'MS $j_{\perp \phi}$',  pklpath, threesixty, axisticks,
                               deltahr)
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax1[2,i], 
                               'MS $j_{\perp Residual}$',  pklpath, threesixty, 
                               axisticks, deltahr)
        
        # We need the filepath for RIM file to find the pickle filename
        # We only search for the nearest minute, ignoring last entry in key
        for key in info['files']['ionosphere']:
            if( key[0] == time[0] and key[1] == time[1] and key[2] == time[2] and \
                key[3] == time[3] and key[4] == time[4] ):
                    filepath = info['files']['ionosphere'][key]
                    
        # filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 

        # Create heatmaps
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax2[0,i], 
                               'Gap $j_\parallel$', pklpath, threesixty, axisticks,
                               deltahr)
        
        # Rinse and repeat for ionosphere
        pklname = basename + '.iono-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
       
        earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax2[1,i], 
                               '$j_{Pederson}$', pklpath, threesixty, axisticks,
                               deltahr)
        im = earth_currents_heatmap( info, time, vmin, vmax, nlat, nlong, ax2[2,i], 
                                    '$j_{Hall}$', pklpath, threesixty, axisticks,
                                    deltahr)

    # Set titles for each column
    for axp, col in zip(ax1[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(time_hhmm)

    for axp, col in zip(ax2[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(time_hhmm)

    for axp, row in zip(ax1[:,0], ['$j_{\parallel}$', \
                                  '$j_{\perp \phi}$', \
                                  '$\Delta j_{\perp}$']):
        axp.set_ylabel(row, rotation=90)
    for axp, row in zip(ax2[:,0], ['Gap $j_{\parallel}$', \
                                  '$j_{P}$', \
                                  '$j_{H}$']):
        axp.set_ylabel(row, rotation=90)

    # Add colorbar
    # cbar1 = fig1.colorbar( im, ax=ax1[3,:], orientation='horizontal')
    # cbar1.set_label(r'$B_{N}$ (nT)')
    # for colp in range(cols): 
    #     fig1.delaxes(ax=ax1[3,colp])
        
    cbar2 = fig2.colorbar( im, ax=ax2[3,:], orientation='horizontal' )
    cbar2.set_label(r'$B_{N}$ (nT)')
    for colp in range(cols): 
        fig2.delaxes(ax=ax2[3,colp])

    # Set titles
    fig1.suptitle(r'$B_{N}$ due to Magnetospheric Currents')
    fig2.suptitle(r'$B_{N}$ due to Gap and Ionospheric Currents')
    
    fig1.tight_layout()
    fig2.tight_layout()
    
    create_directory( info['dir_plots'], 'heatmaps' )
    fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.png" ) )
    fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.png" ) )
    
    # fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.pdf" ) )
    # fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.pdf" ) )
    
    # fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.eps" ) )
    # fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.eps" ) )
    
    # fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.jpg" ) )
    # fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.jpg" ) )
    
    # fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.tif" ) )
    # fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.tif" ) )
    
    # fig1.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-magnetospheric-currents-grid.svg" ) )
    # fig2.savefig( os.path.join( info['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-grid.svg" ) )
    return

#########

def plot_histogram_ms_by_region_grid(info, times, vmin, vmax, binwidth, deltamp, deltabs, 
                                        thicknessns, nearradius, sharex=True, sharey=True,
                                        deltahr=None):
    """Plot heatmaps in a grid, showing Bn contributions from each magnetospheric
    region.

    Inputs:
        info = info on files to be processed, see info = {...} example above
            
        times = the times associated with the files for which we will create
           heatmaps
        
        vmin, vmax = min/max limits of heatmap color scale
    
        binwidth = width of histogram bins
    
        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
            
        threesixty = Boolean, is our map 0->360 or -180->180 in longitude
                    
        sharex, sharey = Boolean, are x and y axis titles shared
                    
        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the plot generated
        
    """

    # Make string to be used below in names, etc.
    params = '[' + str(deltamp) + ',' + str(deltabs) + ',' + str(thicknessns) \
        + ',' + str(nearradius) + ']'

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [17.0,10.0] #[12.8, 10.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })
    
    # One column of plots for each time
    cols = len(times)

    # Create grid of subplots, 4 regions and 1 column for each time in times
    fig, ax = plt.subplots(4,cols, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(right=0.5)
    
    if vmin != None and vmax!= None: 
        bins = range(vmin, vmax, binwidth)
    else:
        bins = binwidth
    
    # Create heatmaps
    for i in range(cols):
        time = times[i]
        
        # We need the filepath for BATSRUS file to get pickle file
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
    
        pklname = basename + '.' + params + '.ms-region-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname)
        
        df = pd.read_pickle(pklpath)

        ax[0,i].hist(df['Magnetosheath'], bins=bins)
        ax[1,i].hist(df['Near Earth'], bins=bins)
        ax[2,i].hist(df['Neutral Sheet'], bins=bins)
        ax[3,i].hist(df['Other'], bins=bins)
        ax[3,i].set_xlabel('$B_N$ Contribution (nT)')
    
    # Add titles to each column
    for axp, col in zip(ax[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(r'$B_{N}$ (nT) ' + time_hhmm)

    # Add titles to each row identifying region
    for axp, row in zip(ax[:,0], ['Magnetosheath\nCounts', 'Near Earth\nCounts', \
                                  'Neutral Sheet\nCounts', 'Other\nCounts']):
        axp.set_ylabel(row, rotation=90)
   
    # Save plot
    create_directory( info['dir_plots'], 'histograms' )
    fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.png" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.pdf" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.eps" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.jpg" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.tif" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-region-grid.svg" ) )
    return

def plot_histogram_ms_by_currents_grid(info, times, vmin, vmax, binwidth, sharex=True, 
                                       sharey=True, deltahr=None):
    """Plot results from loop_heatmapworld_ms, showing histograms of
    Bn contributions from magnetospheric currents.

    Inputs:
       info = info on files to be processed, see info = {...} example above
            
       times = the times associated with the files for which we will create
           heatmaps
        
       vmin, vmax = min/max limits of histogram
       
       binwidth = width of histogram columns
       
       sharex, sharey = Boolean, are x and y axis titles shared
                    
       deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.

    Outputs:
        None - other than the plot generated
        
    """

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [18.0,14.0] #[12.8, 12.0]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 18
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    cols = len(times)
        
    fig, ax = plt.subplots(6, cols, sharex=sharex, sharey=sharey)
    fig.subplots_adjust(right=0.5)
    
    if vmin != None and vmax!= None: 
        bins = range(vmin, vmax, binwidth)
    else:
        bins = binwidth
    
    for i in range(cols):
        time = times[i]
        
        # We need the filepath for BATSRUS file to find the pickle filename
        filepath = info['files']['magnetosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.ms-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
        df = pd.read_pickle(pklpath)
        
        # Create histograms for different currents
        ax[0,i].hist(df[r'Parallel'], bins=bins)
        ax[1,i].hist(df[r'Perpendicular $\phi$'], bins=bins)
        ax[2,i].hist(df[r'Perpendicular Residual'], bins=bins)
        
        # We need the filepath for RIM file to find the pickle filename
        # We only search for the nearest minute, ignoring last entry in key
        for key in info['files']['ionosphere']:
            if( key[0] == time[0] and key[1] == time[1] and key[2] == time[2] and \
                key[3] == time[3] and key[4] == time[4] ):
                    filepath = info['files']['ionosphere'][key]
                    
        # filepath = info['files']['ionosphere'][time]
        basename = os.path.basename(filepath)
        pklname = basename + '.gap-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
        df = pd.read_pickle(pklpath)

        # Create histograms
        ax[3,i].hist(df['Total'], bins=bins)
        
        # Rinse and repeat for ionosphere
        pklname = basename + '.iono-heatmap-world.pkl'
        pklpath = os.path.join( info['dir_derived'], 'heatmaps', pklname) 
        df = pd.read_pickle(pklpath)
      
        ax[4,i].hist(df['Total Pedersen'], bins=bins)
        ax[5,i].hist(df['Total Hall'], bins=bins)
        ax[5,i].set_xlabel('$B_N$ Contribution (nT)')
              
    # Set titles for each column
    for axp, col in zip(ax[0], times):
        if deltahr is None:
            dtime = datetime(*col)
            time_hhmm = dtime.strftime("%H:%M")
        else:
            dtime = datetime(*col) + timedelta(hours=deltahr)
            time_hhmm = dtime.strftime("%H:%M")
        axp.set_title(r'$B_{N}$ (nT) ' + time_hhmm)

    # Set titles for each row
    for axp, row in zip(ax[:,0], ['Magneto-\nsphere $j_{\parallel}$\nCounts', \
                                  'Magneto-\nsphere $j_{\perp \phi}$\nCounts', \
                                  'Magneto-\nsphere $\Delta j_{\perp}$\nCounts', \
                                  'Gap $j_{\parallel}$\nCounts', \
                                  'Ionosphere $j_{P}$\nCounts', \
                                  'Ionosphere $j_{H}$\nCounts']):
        axp.set_ylabel(row, rotation=90)

    create_directory( info['dir_plots'], 'histograms' )
    fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.png" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.pdf" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.eps" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.jpg" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.tif" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'histograms', "histogram-currents-grid.svg" ) )
    return




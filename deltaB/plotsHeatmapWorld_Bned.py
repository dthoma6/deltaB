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
import matplotlib.pyplot as plt
# from matplotlib.colors import SymLogNorm
from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from deltaB import calc_ms_b_paraperp, convert_mhd_to_dataframe, \
    date_timeISO, create_directory, calc_ms_divBint_b, calc_ms_surfint_rCurrents_b, \
    calc_ms_surfint_outer_b

from deltaB.BATSRUS_dataframe import get_batsrus_data_from_cdf
from deltaB.OpenGGCM_dataframe import get_openggcm_data_from_cdf
from deltaB.LFM_dataframe import get_lfm_data_from_cdf

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

def loop_heatmapworldned_ms(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in BATSRUS files to create data for heat maps showing the 
    breakdown of Bn due to currents parallel and perpendicular to B field.  
    Results will be used to generate heatmaps of Bn, Be, Bd from Biot-Savart over 
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
        Bn        = [None] * n
        Bparan    = [None] * n
        Bperpn    = [None] * n
        Bperpphin = [None] * n
        Bperpresn = [None] * n
        Be        = [None] * n
        Bparae    = [None] * n
        Bperpe    = [None] * n
        Bperpphie = [None] * n
        Bperprese = [None] * n
        Bd        = [None] * n
        Bparad    = [None] * n
        Bperpd    = [None] * n
        Bperpphid = [None] * n
        Bperpresd = [None] * n
        Bx        = [None] * n
        By        = [None] * n
        Bz        = [None] * n
        B_lat     = [None] * n
        B_long    = [None] * n
        B_time    = [None] * n

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB heatmap for... {basename}')

        # Read in the MHD file 
        if info['model'] == 'SWMF' or info['model'] == 'BATSRUS':
            mhd = get_batsrus_data_from_cdf(filepath,info)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath)
        elif info['model'] == 'LFM':
            mhd = get_lfm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')

        df = convert_mhd_to_dataframe(mhd)
        
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
                Bn[k], Be[k], Bd[k], Bparan[k], Bparae[k], Bparad[k], \
                        Bperpn[k], Bperpe[k], Bperpd[k], \
                        Bperpphin[k], Bperpphie[k], Bperpphid[k], \
                        Bperpresn[k], Bperprese[k], Bperpresd[k], \
                        Bx[k], By[k], Bz[k]= calc_ms_b_paraperp(X, timeISO, df, northonly=False)
            
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'$B_n$': Bn, 
                            r'Parallel N': Bparan, 
                            r'Perpendicular N': Bperpn, 
                            r'Perpendicular $\phi$ N': Bperpphin, 
                            r'Perpendicular Residual N': Bperpresn,
                            r'$B_e$': Be, 
                            r'Parallel E': Bparae, 
                            r'Perpendicular E': Bperpe, 
                            r'Perpendicular $\phi$ E': Bperpphie, 
                            r'Perpendicular Residual E': Bperprese,
                            r'$B_d$': Bd, 
                            r'Parallel D': Bparad, 
                            r'Perpendicular D': Bperpd, 
                            r'Perpendicular $\phi$ D': Bperpphid, 
                            r'Perpendicular Residual D': Bperpresd,
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time } )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.ms-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, len(times), maxcores)
    logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
    Parallel(n_jobs=num_cores)(delayed(wrap_ms)(p, times, deltahr) for p in range(len(times)))

    return

def loop_heatmapworldned_divB(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in MHD files to create data for heat maps showing the 
    breakdown of Bn due to divB.  Results will be used to generate heatmaps of 
    Bn, Be, Bd from divB integral over surface of earth.

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
    def wrap_divB( p, times, deltahr ):
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

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB heatmap for... {basename}')

        # Read in the MHD file 
        if info['model'] == 'SWMF' or info['model'] == 'BATSRUS':
            mhd = get_batsrus_data_from_cdf(filepath,info)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath)
        elif info['model'] == 'LFM':
            mhd = get_lfm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')

        df = convert_mhd_to_dataframe(mhd)
        
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
    
                # Get the B field at the point X and ISO time using the MHD data
                # results are in SM coordinates
                Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] = calc_ms_divBint_b(X, timeISO, mhd)
            
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'$B_n$': Bn, 
                            r'$B_e$': Be, 
                            r'$B_d$': Bd, 
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time} )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.divB-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, len(times), maxcores)
    logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
    Parallel(n_jobs=num_cores)(delayed(wrap_divB)(p, times, deltahr) for p in range(len(times)))

    return

def loop_heatmapworldned_inner(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in MHD files to create data for heat maps showing the 
    breakdown of Bn due to inner surface integral.  Results will be used to 
    generate heatmaps of Bn, Be, Bd from this integral over surface of earth.

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
    def wrap_inner( p, times, deltahr ):
        # We will walk around the globe collecting B field estimates,
        # the spacing of lat and long samples
        dlat = 180. / nlat
        dlong = 360. / nlong
    
        n = nlat * nlong
    
        # Storage for results
        Bn     = [None] * n
        Be     = [None] * n
        Bd     = [None] * n
        Bx     = [None] * n
        By     = [None] * n
        Bz     = [None] * n
        Birrn  = [None] * n
        Birre  = [None] * n
        Birrd  = [None] * n
        Bsoln  = [None] * n
        Bsole  = [None] * n
        Bsold  = [None] * n
        B_lat  = [None] * n
        B_long = [None] * n
        B_time = [None] * n

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB heatmap for... {basename}')

        # Read in the MHD file 
        if info['model'] == 'SWMF' or info['model'] == 'BATSRUS':
            mhd = get_batsrus_data_from_cdf(filepath,info)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath)
        elif info['model'] == 'LFM':
            mhd = get_lfm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')

        df = convert_mhd_to_dataframe(mhd)
        
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
    
                # Get the B field at the point X and ISO time using the MHD data
                # results are in SM coordinates
                Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] 
                Bn[k], Be[k], Bd[k], Birrn[k], Birre[k], Birrd[k], \
                    Bsoln[k], Bsole[k], Bsold[k], Bx[k], By[k], Bz[k] \
                        = calc_ms_surfint_rCurrents_b(X, timeISO, mhd)
            
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'$B_n$': Bn, 
                            r'$B_e$': Be, 
                            r'$B_d$': Bd, 
                            r'Birrn': Birrn,  
                            r'Birre': Birre,  
                            r'Birrd': Birrd,  
                            r'Bsoln': Bsoln,  
                            r'Bsole': Bsole,  
                            r'Bsold': Bsold,  
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time} )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.inner-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, len(times), maxcores)
    logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
    Parallel(n_jobs=num_cores)(delayed(wrap_inner)(p, times, deltahr) for p in range(len(times)))

    return

def loop_heatmapworldned_outer(info, times, nlat, nlong, deltahr=None, maxcores=20):
    """Loop thru data in MHD files to create data for heat maps showing the 
    breakdown of Bn due to outer surface integral.  Results will be used to 
    generate heatmaps of Bn, Be, Bd from this integral over surface of earth.

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
    def wrap_outer( p, times, deltahr ):
        # We will walk around the globe collecting B field estimates,
        # the spacing of lat and long samples
        dlat = 180. / nlat
        dlong = 360. / nlong
    
        n = nlat * nlong
    
        # Storage for results
        Bn     = [None] * n
        Be     = [None] * n
        Bd     = [None] * n
        Bx     = [None] * n
        By     = [None] * n
        Bz     = [None] * n
        Birrn  = [None] * n
        Birre  = [None] * n
        Birrd  = [None] * n
        Bsoln  = [None] * n
        Bsole  = [None] * n
        Bsold  = [None] * n
        B_lat  = [None] * n
        B_long = [None] * n
        B_time = [None] * n

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[p]]
        basename = os.path.basename(filepath)
    
        logging.info(f'Calculate magnetosphere dB heatmap for... {basename}')

        # Read in the MHD file 
        if info['model'] == 'SWMF' or info['model'] == 'BATSRUS':
            mhd = get_batsrus_data_from_cdf(filepath,info)
        elif info['model'] == 'OpenGGCM':
            mhd = get_openggcm_data_from_cdf(filepath)
        elif info['model'] == 'LFM':
            mhd = get_lfm_data_from_cdf(filepath)
        else:
            import sys
            sys.exit(f'Unknown model type: {info["model"]}')

        df = convert_mhd_to_dataframe(mhd)
        
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
    
                # Get the B field at the point X and ISO time using the MHD data
                # results are in SM coordinates
                Bn[k], Be[k], Bd[k], Bx[k], By[k], Bz[k] 
                Bn[k], Be[k], Bd[k], Birrn[k], Birre[k], Birrd[k], \
                    Bsoln[k], Bsole[k], Bsold[k], Bx[k], By[k], Bz[k] \
                        = calc_ms_surfint_outer_b(X, timeISO, mhd)
            
        # Put the results in a dataframe and save it.
        df = pd.DataFrame( { r'$B_n$': Bn, 
                            r'$B_e$': Be, 
                            r'$B_d$': Bd, 
                            r'Birrn': Birrn,  
                            r'Birre': Birre,  
                            r'Birrd': Birrd,  
                            r'Bsoln': Bsoln,  
                            r'Bsole': Bsole,  
                            r'Bsold': Bsold,  
                            r'Latitude': B_lat,
                            r'Longitude': B_long, 
                            r'Time': B_time} )
        
        create_directory(info['dir_derived'], 'heatmaps')
        pklname = basename + '.outer-heatmap-world.pkl'
        df.to_pickle( os.path.join( info['dir_derived'], 'heatmaps', pklname) )

    # Make sure deltahr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Loop through the files using parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    num_cores = min(num_cores, len(times), maxcores)
    logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
    Parallel(n_jobs=num_cores)(delayed(wrap_outer)(p, times, deltahr) for p in range(len(times)))

    return


def earth_helmholtz_heatmap( info, time, vmin, vmax, nlat, nlong, ax, title,  
                         threesixty, axisticks, deltahr, integral='ms'):   
    """Plot results from loop_heatmapworld_..., showing the heatmap of
    Bn, Be, Bd contributions the Biot-Savart integral from Helmholtz Decompostion
    contributes at a specific time

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
            
        integral = which Helmholtz integral. 'ms' (aka Biot-Savart), 'divB', 
            'outer', or 'inner'

    Outputs:
        im = pseudocolor color plot
           - the plot generated
        
    """
    # We need the filepath for BATSRUS file to get pickle file
    filepath = info['files']['magnetosphere'][time]
    basename = os.path.basename(filepath)

    if integral == 'ms':
        pklname = basename + '.ms-heatmap-world.pkl'
    elif integral == 'divB':
        pklname = basename + '.divB-heatmap-world.pkl'
    elif integral == 'inner':
        pklname = basename + '.inner-heatmap-world.pkl'
    elif integral == 'outer':
        pklname = basename + '.outer-heatmap-world.pkl'
    else:
        assert False
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

def plot_heatmapworld_helmholtz_grid(info, times, vmin, vmax, nlat, nlong,
                                        threesixty=False, axisticks=False,
                                        deltahr=None, component='r$B_n$'):
    """Plot heatmaps in a grid, showing Bn, Be, Bd contributions from each 
    Helmholtz Decompostion integral.

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
            
        component = dataframe variable used to color heatmap, e.g., r'$B_n$'

    Outputs:
        None - other than the plot generated
        
    """
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
         
        earth_helmholtz_heatmap( info, time, vmin, vmax, nlat, nlong, ax[0,i], 
                             component, threesixty, axisticks, deltahr, integral='ms')
        earth_helmholtz_heatmap( info, time, vmin, vmax, nlat, nlong, ax[1,i], 
                             component, threesixty, axisticks, deltahr, integral='inner')
        earth_helmholtz_heatmap( info, time, vmin, vmax, nlat, nlong, ax[2,i],
                              component, threesixty, axisticks, deltahr, integral='divB')
        im = earth_helmholtz_heatmap( info, time, vmin, vmax, nlat, nlong, ax[3,i], 
                              component, threesixty, axisticks, deltahr, integral='outer')
    
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
    for axp, row in zip(ax[:,0], [r'$\mathsf{B_{BS}}$', 
                                  r'$\mathsf{B_{HDT}}$', 
                                  r'$\mathsf{\delta B_{div}}$', 
                                  r'$\mathsf{\delta B_{out}}$']):
        axp.set_ylabel(row, rotation=90)
   
    # Add colorbar
    cbar = fig.colorbar( im, ax=ax[4,:], orientation='horizontal' )
    if component == r'$B_n$': component = r'$\mathsf{B_N}$'
    if component == r'$B_e$': component = r'$\mathsf{B_E}$'
    if component == r'$B_d$': component = r'$\mathsf{B_D}$'
    cbar.set_label(component +' (nT)')
    for colp in range(cols): 
        fig.delaxes(ax=ax[4,colp])

    # Set title
    fig.suptitle(component + ' due to Helmholtz Decomposition')

    # Save plot
    create_directory( info['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.png" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.pdf" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.eps" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.jpg" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.tif" ) )
    # fig.savefig( os.path.join( info['dir_plots'], 'heatmaps', component + "-heatmap-helmholtz-grid.svg" ) )
    return



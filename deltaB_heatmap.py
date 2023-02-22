#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:45:41 2023

@author: Dean Thomas
"""

import logging
import pandas as pd

from spacepy import coordinates as coord
from spacepy.time import Ticktock

from deltaB.BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_cumulative_sum_dataframe
from deltaB.util import ned, date_timeISO, create_directory

COLABA = True

# origin and target define where input data and output plots are stored
if COLABA:
    ORIGIN = '/Volumes/Physics HD v2/runs/DIPTSUR2/GM/IO2/'
    TARGET = '/Volumes/Physics HD v2/runs/DIPTSUR2/'
else:
    ORIGIN = '/Volumes/Physics HD v2/divB_simple1/GM/'
    TARGET = '/Volumes/Physics HD v2/divB_simple1/'

# rCurrents define range from earth center below which results are not valid
# measured in Re units
if COLABA:
    RCURRENTS = 1.8
else:
    RCURRENTS = 3

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

def process_sum_db_at_X(X, timeiso, df):
    """Process data in BATSRUS file to create dataframe with calculated quantities

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        timeiso = time in ISO format -> '2002-02-25T12:20:30'
        
        df = dataframe with BATRUS data

     Outputs:
        dBNSum1 = north component of total B field at point X
        
        dBparallelNSum1 = north component of B field at point X due to currents
            parallel to B field
            
        dBperpendicularNSum1 = north component of B field at point X due to currents
            penpendicular to B field
            
        dBperpendicularphiNSum1 = north component of B field at point X due to currents
            perpendicular to B field and in phi-hat direction (j_perpendicular dot phi-hat)
            
        dBperpendicularphiresNSum1 = = north component of B field at point X due to 
            residual currents perpendicular to B field (j_perpendicular minus 
            j_perpendicular dot phi-hat)
    """

    logging.info('Calculate delta B...')

    df1 = create_deltaB_rCurrents_dataframe(df, X)

    logging.info('Calculate cumulative sums...')

    df1 = create_cumulative_sum_dataframe(df1)

    # Get north-east-down unit vectors at point X 
    n_geo, e_geo, d_geo = ned(timeiso, X, 'GSM')
    
    def north_comp( df, n_geo ):
        """ Local function used to get north component of field defined in df.
        
        Inputs:
            df = dataframe with magnetic field info
            n_geo = north unit vector
        Outputs:
            dBNSum = Total B north component
            dBparallelNSum = Total B due to currents parallel to B field, 
                north component
            dBperpendicularNSum = Total B due to currents perpendicular to B 
                field, north component
            dBperpendicularphiNSum = Total B due to currents perpendicular to B 
                field, north component, but divided into a piece along phi-hat
                and the residual
            dBperpendicularphiresNSum =Total B due to currents perpendicular to B 
                field, north component, but divided into a piece along phi-hat
                and the residual
        """
        dBNSum = df['dBxSum'].iloc[-1]*n_geo[0] + \
            df['dBySum'].iloc[-1]*n_geo[1] + \
            df['dBzSum'].iloc[-1]*n_geo[2]
        dBparallelNSum = df['dBparallelxSum'].iloc[-1]*n_geo[0] + \
            df['dBparallelySum'].iloc[-1]*n_geo[1] + \
            df['dBparallelzSum'].iloc[-1]*n_geo[2]
        dBperpendicularNSum = df['dBperpendicularxSum'].iloc[-1]*n_geo[0] + \
            df['dBperpendicularySum'].iloc[-1]*n_geo[1] + \
            df['dBperpendicularzSum'].iloc[-1]*n_geo[2]
        dBperpendicularphiNSum = df['dBperpendicularphixSum'].iloc[-1]*n_geo[0] + \
            df['dBperpendicularphiySum'].iloc[-1]*n_geo[1] + \
            df['dBperpendicularphizSum'].iloc[-1]*n_geo[2]
        dBperpendicularphiresNSum = df['dBperpendicularphiresxSum'].iloc[-1]*n_geo[0] + \
            df['dBperpendicularphiresySum'].iloc[-1]*n_geo[1] + \
            df['dBperpendicularphireszSum'].iloc[-1]*n_geo[2]
        return dBNSum, dBparallelNSum, dBperpendicularNSum, dBperpendicularphiNSum, \
            dBperpendicularphiresNSum
            
    dBNSum1, dBparallelNSum1, dBperpendicularNSum1, dBperpendicularphiNSum1, \
        dBperpendicularphiresNSum1 = north_comp( df1, n_geo )

    return dBNSum1, \
        dBparallelNSum1, \
        dBperpendicularNSum1, \
        dBperpendicularphiNSum1, \
        dBperpendicularphiresNSum1

def loop_sum_db(base, dirpath, tgtpath, nlat, nlong, rCurrents):
    """Loop thru data in BATSRUS files to create plots showing the breakdown of
    parallel and perpendicular to B field components

    Inputs:
        
        base, dirpath = data read from BATSRUS file is at dirpath + base + .out 
        
        tgtpath = path to folder when data will be saved in a subdirectory 'heatmaps'
        
        nlat, nlong = number of latitude and longitude samples
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
    Outputs:
        None - other than the pickle file that is generated
    """

    # We will walk around the globe collecting B field estimates,
    # the spacing of lat and long samples
    dlat = 180. / nlat
    dlong = 360. / nlong

    n = nlat * nlong
    
    # Storage for results
    b_original = [None] * n
    b_original_parallel = [None] * n
    b_original_perp = [None] * n
    b_original_perpphi = [None] * n
    b_original_perpphires = [None] * n
    b_lat = [None] * n
    b_long = [None] * n
    b_time = [None] * n

    # Read in the BATSRUS file 
    filename = dirpath + base
    df = convert_BATSRUS_to_dataframe(filename, rCurrents=rCurrents)
    
    # Get the ISO time from the filename (base)
    timeiso = date_timeISO(base)
    
    # Determine where noon is, that is longitude for (1,0,0) GSM
    noongsm = coord.Coords([[1,0,0]], 'GSM', 'car')
    noongsm.ticks = Ticktock([timeiso], 'ISO')
    noongeo = noongsm.convert('GEO', 'sph')
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
            b_lat[k] = 90. - (i + 0.5)*dlat
            long = noonlatlong[2] + 180. - (j + 0.5)*dlong
            if( long > 180 ): long = long - 360
            if( long < -180 ): long = long + 360
            b_long[k] = long
            b_time[k] = (j + 0.5) * 24. / nlong
            
            # We need to convert the lat-long into GSM coordiantes for use
            # with BATSRUS data.  Our point is on the earth's surface, so the
            # first entry is 1.
            Xlatlong=[1., b_lat[i*nlong + j], b_long[i*nlong + j]]
            Xgeo = coord.Coords([Xlatlong], 'GEO', 'sph')
            Xgeo.ticks = Ticktock([timeiso], 'ISO')
            Xgsm = Xgeo.convert('GSM', 'car')
            X = Xgsm.data[0]

            # Get the B field at the point X and timeiso using the BATSRUS data
            # in df
            b_original[k], b_original_parallel[k], b_original_perp[k], \
                b_original_perpphi[k], b_original_perpphires[k] = \
                process_sum_db_at_X(X, timeiso, df)
    
    # Determine the fraction of the B field due to various currents - those
    # parallell to the B field, perpendicular to B field and in the phi-hat
    # direction (jperpenddicular dot phi-hat), and the remaining perpendicular 
    # current (jperpendicular - jperpendicular dot phi-hat).
    b_fraction_parallel = [m/n for m, n in zip(b_original_parallel, b_original)]
    b_fraction_perp = [m/n for m, n in zip(b_original_perp, b_original)]
    b_fraction_perpphi = [m/n for m, n in zip(b_original_perpphi, b_original)]
    b_fraction_perpphires = [m/n for m, n in zip(b_original_perpphires, b_original)]

    # Put the results in a dataframe and save it.
    df = pd.DataFrame( { r'Total': b_original, 
                        r'Parallel': b_original_parallel, 
                        r'Perpendicular': b_original_perp, 
                        r'Perpendicular $\phi$': b_original_perpphi, 
                        r'Perpendicular Residual': b_original_perpphires,
                        r'Latitude': b_lat,
                        r'Longitude': b_long, 
                        r'Time': b_time,
                        r'Fraction Parallel': b_fraction_parallel, 
                        r'Fraction Perpendicular': b_fraction_perp, 
                        r'Fraction Perpendicular $\phi$': b_fraction_perpphi, 
                        r'Fraction Perpendicular Residual': b_fraction_perpphires } )
    
    create_directory( tgtpath, 'heatmaps/' )
    df.to_pickle( tgtpath + 'heatmaps/' + base + '.heatmap.pkl')
    
    return

if __name__ == "__main__":
    rCurrents = RCURRENTS
    dirpath = ORIGIN
    tgtpath = TARGET
   
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    cmap = plt.colormaps['coolwarm']

    
    # bases = ('3d__var_2_e20190902-041500-031',
    #           '3d__var_2_e20190902-053000-000',
    #           '3d__var_2_e20190902-060000-021',
    #           '3d__var_2_e20190902-063000-000',
    #           '3d__var_2_e20190902-103000-000' )
    
    bases = ('3d__var_2_e20190902-041500-031',
              '3d__var_2_e20190902-063000-000',
              '3d__var_2_e20190902-103000-000' )
    
    nlat = 9
    nlong = 12
    
    for i in range(len(bases)):
    # for i in [1]:
        base= bases[i]
        
        # loop_sum_db(base, dirpath, tgtpath, nlat, nlong, rCurrents)
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        from deltaB.util import date_time
        
        vmin = -1500.
        vmax = 1500.
        
        y, m, d, hh, mm, ss = date_time(base)
        # time = str(hh*100+mm)
        time = str(hh) + ':' + str(mm)
        
        df = pd.read_pickle(tgtpath + 'heatmaps/' + base + '.heatmap.pkl')
    
        # Earth rotates CCW viewed from above North Pole.  Dawn happens on west 
        # side of globe as the earth rotates to the east.
        # https://commons.wikimedia.org/wiki/File:AxialTiltObliquity.png
        
        # set_xticks has a trick.  The ticks are numbered by the columns in the
        # pivot table, not by the values of the xaxis.
        
        df1 = df.pivot(index='Latitude', columns='Time', values='Total' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'$B_N(\mathbf{j})$ ' + f'({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', 'Parallel' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\parallel)}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp)}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular $\phi$' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp \cdot \hat \phi)}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Perpendicular Residual' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi)}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', 'Fraction Parallel' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=-1., vmax=1.)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\parallel)}/{B_N(\mathbf{j})}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Fraction Perpendicular' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=-1., vmax=1.)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp)}/{B_N(\mathbf{j})}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Fraction Perpendicular $\phi$' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=-1., vmax=1.)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp \cdot \hat \phi)}/{B_N(\mathbf{j})}$' + f' ({time})')
        plt.show()
        
        df1 = df.pivot('Latitude', 'Time', r'Fraction Perpendicular Residual' )
        df1 = df1.sort_values('Latitude',ascending=False)
        ax = sns.heatmap(df1, cmap=cmap, vmin=-1., vmax=1.)
        ax.invert_xaxis()
        ax.set_xticks([0,3,6,9,12])
        ax.set_xticklabels(['24:00','18:00','12:00','06:00','24:00'])
        ax.set_xlabel('Local Time')
        plt.title(r'${B_N(\mathbf{j}_\perp -\mathbf{j}_\perp \cdot \hat \phi)}/{B_N(\mathbf{j})}$' + f' ({time})')
        plt.show()

    
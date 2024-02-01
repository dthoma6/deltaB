#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 19:22:28 2024

@author: Dean Thomas
"""

import os.path

############################################################################
#
# Script to produce figure in paper that shows the similarties between
# the Scenario 1 and Scenario 2 heatmaps
#
# compare_heatmaps is the workhorse. It is derived from 
# plot_heatmapworld_ms_by_currents_grid2 in plotsHeatmapWorld.py, which is
# used to generate heatmaps in other parts of the paper.
#
############################################################################

# Where are the data files

data_dir = r'/Volumes/PhysicsHDv3'

CARR_info = {
        "model": "SWMF",
        "run_name": "CARR_Scenario1",
        "rCurrents": 1.8,
        "rIonosphere": 1.01725,
        "file_type": "out",
        "dir_run": os.path.join(data_dir, "CARR_Scenario1"),
        "dir_plots": os.path.join(data_dir, "CARR_Scenario1.plots"),
        "dir_derived": os.path.join(data_dir, "CARR_Scenario1.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "CARR_Scenario1", "MAG-3D"),
        "dir_ionosphere": os.path.join(data_dir, "CARR_Scenario1", "IONO-2D")
}

Chigo_info = {
        "model": "SWMF",
        "run_name": "Chigomezyo_Ngwira_092112_3a",
        "rCurrents": 1.5,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a"),
        "dir_plots": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a.plots"),
        "dir_derived": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a", "GM_CDF"),
        "dir_ionosphere": os.path.join(data_dir, "Chigomezyo_Ngwira_092112_3a", "IONO-2D_CDF")
}

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# Colormap used in heatmaps below
COLORMAP = 'coolwarm'

from deltaB import find_regions, calc_ms_b_paraperp, calc_ms_b_region,\
    calc_iono_b, calc_gap_b, calc_gap_b_rim, \
    convert_BATSRUS_to_dataframe, \
    earth_currents_heatmap, \
    date_timeISO, create_directory

# Parameters borrowed from the scripts to generate the other heatmaps
CARRVMIN = -2000
CARRVMAX = +2000

ChigoVMIN = -2500
ChigoVMAX = +2500

NLAT = 30
NLONG = 60

CARRtime = (2019, 9, 2, 6, 30, 0)
Chigotime = (2003, 9, 2, 2, 0, 0)

DELTAHR = 5.5

# Code borrowed from plotHeatmapsWorld.py, this routine does the heavy lifting

def compare_heatmaps(CARRinfo, Chigoinfo, CARRtime, Chigotime, CARRvmin, CARRvmax, 
                                          Chigovmin, Chigovmax, nlat, nlong, 
                                          threesixty = False, axisticks = False,
                                          deltahr=None):
    """Compare heatmaps from CARR and Chigo scenarios to demonstrate similarities

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
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 12
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Helvetica",
    })

    cols = 2
    
    # Is our map 0->360 or -180->180
    if threesixty:
        proj = ccrs.PlateCarree(central_longitude=180.)
    else:
        proj = ccrs.PlateCarree()
    
    # Create fig1 for magnetospheric currents and fig2 for gap & ionospheric currents
    plt.rcParams["figure.figsize"] = [5.0, 4.0] #[7.0,6.1] #[7.0,8.0]
    fig, ax = plt.subplots(3, cols, sharex=True, sharey=True, subplot_kw={'projection': proj})
        
    # We need the filepath for RIM file to find the pickle filename
    # We only search for the nearest minute, ignoring last entry in key
    for key in CARRinfo['files']['ionosphere']:
        if( key[0] == CARRtime[0] and key[1] == CARRtime[1] and key[2] == CARRtime[2] and \
            key[3] == CARRtime[3] and key[4] == CARRtime[4] ):
                CARRfilepath = CARRinfo['files']['ionosphere'][key]
                
    for key in Chigoinfo['files']['ionosphere']:
        if( key[0] == Chigotime[0] and key[1] == Chigotime[1] and key[2] == Chigotime[2] and \
            key[3] == Chigotime[3] and key[4] == Chigotime[4] ):
                Chigofilepath = Chigoinfo['files']['ionosphere'][key]
                
     # filepath = info['files']['ionosphere'][time]
    CARRbasename = os.path.basename(CARRfilepath)
    CARRpklname = CARRbasename + '.gap-heatmap-world.pkl'
    CARRpklpath = os.path.join( CARRinfo['dir_derived'], 'heatmaps', CARRpklname) 

    Chigobasename = os.path.basename(Chigofilepath)
    Chigopklname = Chigobasename + '.gap-heatmap-world.pkl'
    Chigopklpath = os.path.join( CARRinfo['dir_derived'], 'heatmaps', Chigopklname) 

    # Create heatmaps
    earth_currents_heatmap( CARRinfo, CARRtime, CARRvmin, CARRvmax, nlat, nlong, ax[0,0], 
                           'Gap $j_\parallel$', CARRpklpath, threesixty, axisticks,
                           None)
    
    earth_currents_heatmap( Chigoinfo, Chigotime, Chigovmin, Chigovmax, nlat, nlong, ax[0,1], 
                           'Gap $j_\parallel$', CARRpklpath, threesixty, axisticks,
                           deltahr)
   
    # Rinse and repeat for ionosphere
    CARRpklname = CARRbasename + '.iono-heatmap-world.pkl'
    CARRpklpath = os.path.join( CARRinfo['dir_derived'], 'heatmaps', CARRpklname) 
   
    Chigopklname = Chigobasename + '.iono-heatmap-world.pkl'
    Chigopklpath = os.path.join( Chigoinfo['dir_derived'], 'heatmaps', Chigopklname) 
   
    earth_currents_heatmap( CARRinfo, CARRtime, CARRvmin, CARRvmax, nlat, nlong, ax[1,0], 
                           '$j_{Pederson}$', CARRpklpath, threesixty, axisticks,
                           None)
    earth_currents_heatmap( CARRinfo, CARRtime, CARRvmin, CARRvmax, nlat, nlong, ax[2,0], 
                            '$j_{Hall}$', CARRpklpath, threesixty, axisticks,
                            None)

    earth_currents_heatmap( Chigoinfo, Chigotime, Chigovmin, Chigovmax, nlat, nlong, ax[1,1], 
                           '$j_{Pederson}$', Chigopklpath, threesixty, axisticks,
                           deltahr)
    earth_currents_heatmap( Chigoinfo, Chigotime, Chigovmin, Chigovmax, nlat, nlong, ax[2,1], 
                            '$j_{Hall}$', Chigopklpath, threesixty, axisticks,
                            deltahr)

    # Set titles for each column
    dtime = datetime(*CARRtime)
    time_hhmm = dtime.strftime("%H:%M")
    ax[0,0].set_title('Scenario 1: ' + time_hhmm)

    dtime = datetime(*Chigotime) + timedelta(hours=deltahr)
    time_hhmm = dtime.strftime("%H:%M")
    ax[0,1].set_title('Scenario 2: ' + time_hhmm)

    for axp, row in zip(ax[:,0], ['Gap $j_{\parallel}$', \
                                  '$j_{P}$', \
                                  '$j_{H}$']):
        axp.set_ylabel(row, rotation=90)

    # Add colorbar
    # cbar1 = fig2.colorbar( CARRim, ax=ax2[3,0], orientation='horizontal' )
    # cbar1.set_label(r'$B_{N}$ (nT)')
    # cbar2 = fig2.colorbar( Chigoim, ax=ax2[3,1], orientation='horizontal' )
    # cbar2.set_label(r'$B_{N}$ (nT)')
    # for colp in range(cols): 
    #     fig2.delaxes(ax=ax2[3,colp])

    # Set titles
    fig.suptitle(r'$B_{N}$ due to Gap and Ionospheric Currents')
    
    create_directory( CARRinfo['dir_plots'], 'heatmaps' )
    fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.png" ) )
    # fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.pdf" ) )
    # fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.eps" ) )
    # fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.jpg" ) )
    # fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.tif" ) )
    # fig.savefig( os.path.join( CARRinfo['dir_plots'], 'heatmaps', "heatmap-gap&ionospheric-currents-compare.svg" ) )
    return

###################################################
# Code taken from magnetopost.  Modified to handle this situation.  
# We have cdf files, not .out files.  No _GM_cdf_list file. Nor
# any ionosphere files.
###################################################

def Chigosetup(info):
    # import magnetopost as mp

    assert os.path.exists(info["dir_run"]), "dir_run = " + info["dir_run"] + " not found"

    dir_steps = os.path.join(info["dir_derived"], "timeseries", "timesteps")

    if not os.path.exists(info["dir_plots"]):
        os.mkdir(info["dir_plots"])
        # mp.logger.info("Created " + info["dir_plots"])

    if not os.path.exists(info["dir_derived"]):
        os.mkdir(info["dir_derived"])
        # mp.logger.info("Created " + info["dir_derived"])
    
    if not os.path.exists(dir_steps):
        os.makedirs(os.path.join(dir_steps))
        # mp.logger.info("Created " + dir_steps)

    info['files'] = {}

    # if info['file_type'] == 'out':
    if info['file_type'] == 'cdf':

        generate_filelist_txts(info)

        info['files']['magnetosphere'] = {}
        with open(os.path.join(info["dir_derived"], 'magnetosphere_files.txt'), 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                time = tuple([int(ti) for ti in items[:6]])
                info['files']['magnetosphere'][time] = os.path.join(info['dir_run'], items[-1][:-1])

        info['files']['ionosphere'] = {}
        with open(os.path.join(info["dir_derived"], 'ionosphere_files.txt'), 'r') as f:
            for line in f.readlines():
                items = line.split(' ')
                time = tuple([int(ti) for ti in items[:6]])
                info['files']['ionosphere'][time] = os.path.join(info['dir_run'], items[-1][:-1])

def generate_filelist_txts(info):

    import os
    import re
    import json

    # import magnetopost as mp

    dir_run = info["dir_run"]

    fn = os.path.join(info["dir_derived"], 'run.info.py')
    with open(fn, 'w') as outfile:
        outfile.write(json.dumps(info))

    # mp.logger.info("Wrote {}".format(fn))

    if 'dir_magnetosphere' in info:
        dir_data = os.path.join(dir_run, info['dir_magnetosphere'])
    else:
        dir_data = os.path.join(dir_run, 'GM/IO2')

    magnetosphere_outs = sorted(os.listdir(dir_data))

    fn = os.path.join(info["dir_derived"], 'magnetosphere_files.txt')
    # fn =   os.path.join(info['dir_run'], 'GM_CDF', info['run_name'] + '_GM_cdf_list')

    k = 0
    with open(fn,'w') as fl:
        # regex = r"3d__.*\.out$"
        regex = r"3d__.*\.out.cdf$"
        for fname in magnetosphere_outs:
            if re.search(regex, fname):
                k = k + 1
                assert(fname[:4] == '3d__')
                assert(fname[9] == '_')
                if fname[10] == 'e':
                    Y = int(fname[11:15])
                    M = int(fname[15:17])
                    D = int(fname[17:19])
                    assert(fname[19] == '-')
                    h = int(fname[20:22])
                    m = int(fname[22:24])
                    s = int(fname[24:26])
                    assert(fname[26] == '-')
                    mil = int(fname[27:30])
                    # assert(fname[30:] == '.out')
                    assert(fname[30:] == '.out.cdf')
                    fl.write(f'{Y} {M} {D} {h} {m} {s} {mil} {dir_data}/{fname}\n')
                    # entry = fname + '  Date: ' + str(Y).zfill(4) + '/' + str(M).zfill(2) + '/' + \
                    #     str(D).zfill(2) + ' Time: ' + str(h).zfill(2) +':' + str(m).zfill(2) + \
                    #     ':' + str(s).zfill(2) +'\n'
                    # fl.write(entry)

    # mp.logger.info("Wrote {} file names to {}".format(k, fn))

    if 'dir_ionosphere' in info:
        dir_data = os.path.join(dir_run, info['dir_ionosphere'])
    else:
        dir_data = os.path.join(dir_run, 'IE/ionosphere')

    ionosphere_outs = sorted(os.listdir(dir_data))

    fn = os.path.join(info["dir_derived"], 'ionosphere_files.txt')

    k = 0
    with open(fn,'w') as fl:
        regex = r"null.swmf.i_e.*\.cdf$"

        for fname in ionosphere_outs:
            if re.search(regex, fname):
                k = k + 1

                    # null.swmf.i_e20120723-174500-000.cdf
                    # 012345678901234567890123456789012345

                # assert(fname[:2] == 'it')
                assert(fname[:13] == 'null.swmf.i_e')
                if fname[0] == 'n':
                    Y = int(fname[13:17])
                    M = int(fname[17:19])
                    D = int(fname[19:21])
                    assert(fname[21] == '-')
                    h = int(fname[22:24])
                    m = int(fname[24:26])
                    s = int(fname[26:28])
                    assert(fname[28] == '-')
                    mil = int(fname[29:32])
                    assert(fname[32:] == '.cdf')
                    fl.write(f'{Y} {M} {D} {h} {m} {s} {mil} {dir_data}/{fname}\n')

    # mp.logger.info("Wrote {} file names to {}".format(k, fn))

###############

if __name__ == "__main__":

    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(CARR_info)
    Chigosetup(Chigo_info)
    
    compare_heatmaps(CARR_info, Chigo_info, CARRtime, Chigotime, CARRVMIN, CARRVMAX, 
                                              ChigoVMIN, ChigoVMAX, NLAT, NLONG, 
                                              threesixty = False, axisticks = False,
                                              deltahr=DELTAHR)
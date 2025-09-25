#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:08:36 2024

@author: Dean Thomas
"""

import pandas as pd
import os.path
import matplotlib.pyplot as plt
from deltaB import create_directory, get_batsrus_data_from_cdf, get_openggcm_data_from_cdf, \
    BATSRUS_curlBtoJ, OpenGGCM_curlBtoJ
from datetime import datetime
import numpy as np
from copy import deepcopy


from Bob_Weigel_070323_3_info import info as bwinfo

point = 'Colaba'

bwrootdir = '/Volumes/PhysicsHD/Bob_Weigel_070323_3.derived/timeseries_line_current'
dtrootdir = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived/timeseries_line_current'

bwdata_dir = r'/Volumes/PhysicsHD'
dtdata_dir = r'/Volumes/PhysicsHD'

bwinfo = {
        "model": "SWMF",
        "run_name": "Bob_Weigel_070323_3",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "method": "method1",
        "dir_run": os.path.join(dtdata_dir, "Bob_Weigel_070323_3"),
        "dir_plots": os.path.join(dtdata_dir, "Bob_Weigel_070323_3.plots"),
        "dir_derived": os.path.join(dtdata_dir, "Bob_Weigel_070323_3.derived"),
        "dir_magnetosphere": os.path.join(dtdata_dir, "Bob_Weigel_070323_3", "GM_CDF"),
        "dir_ionosphere": os.path.join(dtdata_dir, "Bob_Weigel_070323_3", "IONO-2D_CDF")
}
dtinfo = {
        "model": "OpenGGCM",
        "run_name": "Dean_Thomas_052924_1",
        # "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(dtdata_dir, "Dean_Thomas_052924_1"),
        "dir_plots": os.path.join(dtdata_dir, "Dean_Thomas_052924_1.plots"),
        "dir_derived": os.path.join(dtdata_dir, "Dean_Thomas_052924_1.derived"),
        "dir_magnetosphere": os.path.join(dtdata_dir, "Dean_Thomas_052924_1", "GM_CDF"),
        "dir_ionosphere": os.path.join(dtdata_dir, "Dean_Thomas_052924_1", "IONO-2D_CDF")
}

bwfile = '/Volumes/PhysicsHD/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-001800-000.out.cdf'
dtfile = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.023400.cdf'

# Create diretory for plots
create_directory( bwinfo['dir_plots'], 'timeseries_line_current' )

batsdata = get_batsrus_data_from_cdf(bwfile, bwinfo)
oggcmdata = get_openggcm_data_from_cdf(dtfile, dtinfo)

# Create a magnetic field due to a line current parallel to x-axis
# offset 2*yGlobalMax in y-direction (So curl and div are zero inside volume)

batsdata2 = deepcopy(batsdata.data_arr)
yGlobalMax = batsdata.yGlobalMax
data_arr = batsdata.data_arr
varidx = batsdata.varidx

# rho squared around x-axis
rho2 = ( data_arr[:, varidx['y']] + 2*yGlobalMax )**2 + data_arr[:, varidx['z']]**2

# New magnetic field
data_arr[:, varidx['bx']] = 0.
data_arr[:, varidx['by']] = - data_arr[:, varidx['z']] / rho2 # by = - sin(phi)/rho
data_arr[:, varidx['bz']] = + (data_arr[:, varidx['y']] + 2*yGlobalMax ) / rho2 # bz = cos(phi)/rho

# New field mean magnitude
Bnew = np.mean( np.sqrt(data_arr[:, varidx['bx']]**2 
                        + data_arr[:, varidx['by']]**2 
                        + data_arr[:, varidx['bz']]**2) )

# Normalize field to have a mean magnitude of Bmag
Bmag = 20.0
data_arr[:, varidx['bx']] = data_arr[:, varidx['bx']] * Bmag / Bnew
data_arr[:, varidx['by']] = data_arr[:, varidx['by']] * Bmag / Bnew
data_arr[:, varidx['bz']] = 0.

# Use curlB to determine current density, j, rather than use OpenGGCM 
# provided values

DataArray = batsdata.DataArray
nVar = batsdata.DataArray.shape[0]
nI = batsdata.nI
nJ = batsdata.nJ
nK = batsdata.nK
nBlock = batsdata.nBlock
rCurrents = batsdata.rCurrents

data_arr = BATSRUS_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, nBlock, rCurrents)

# Create a magnetic field due to a line current parallel to x-axis
# offset 2*yGlobalMax in y-direction (So curl and div are zero inside volume)

oggcmdata2 = deepcopy(oggcmdata.data_arr)
yGlobalMaxGSE = oggcmdata.yGlobalMaxGSE
data_arr = oggcmdata.data_arr
varidx = oggcmdata.varidx

# rho squared around x-axis
rho2 = ( data_arr[:, varidx['y']] + 2*yGlobalMaxGSE )**2 + data_arr[:, varidx['z']]**2

# New magnetic field
data_arr[:, varidx['bx']] = 0.
data_arr[:, varidx['by']] = - data_arr[:, varidx['z']] / rho2 # by = - sin(phi)/rho
data_arr[:, varidx['bz']] = + (data_arr[:, varidx['y']] + 2*yGlobalMaxGSE ) / rho2 # bz = cos(phi)/rho

# New field mean magnitude
Bnew = np.mean( np.sqrt(data_arr[:, varidx['bx']]**2 
                        + data_arr[:, varidx['by']]**2 
                        + data_arr[:, varidx['bz']]**2) )

# Normalize field to have a mean magnitude of Bmag
Bmag = 20.0
data_arr[:, varidx['bx']] = data_arr[:, varidx['bx']] * Bmag / Bnew
data_arr[:, varidx['by']] = data_arr[:, varidx['by']] * Bmag / Bnew
data_arr[:, varidx['bz']] = 0.

# Use curlB to determine current density, j, rather than use OpenGGCM 
# provided values

DataArray = oggcmdata.DataArray
nVar = oggcmdata.DataArray.shape[0]
nI = oggcmdata.nI
nJ = oggcmdata.nJ
nK = oggcmdata.nK
rCurrents = oggcmdata.rCurrents

data_arr = OpenGGCM_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, rCurrents)

# Set some plot configs

plt.rcParams["figure.figsize"] = [12,4]
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12
plt.rcParams.update({
    # "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

# Create plots and save them 

MINMAX = 0.05
BINS = 10
fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].hist(batsdata2[:,batsdata.varidx['jx']], bins=BINS, label=r'$j_x$')
ax[1].hist(batsdata2[:,batsdata.varidx['jy']], bins=BINS, label=r'$j_y$')
ax[2].hist(batsdata2[:,batsdata.varidx['jz']], bins=BINS, label=r'$j_z$')

ax[0].set_xlabel(r'Real $j_x$')
ax[1].set_xlabel(r'Real $j_y$')
ax[2].set_xlabel(r'Real $j_z$')

# ax[0].set_xlim([-MINMAX,MINMAX])
# ax[1].set_xlim([-MINMAX,MINMAX])
# ax[2].set_xlim([-MINMAX,MINMAX])

print( 'BATS jx real mean (std):', np.mean(batsdata2[:,batsdata.varidx['jx']]), ' (', np.std(batsdata2[:,batsdata.varidx['jx']]), ')')
print( 'BATS jy real mean (std):', np.mean(batsdata2[:,batsdata.varidx['jy']]), ' (', np.std(batsdata2[:,batsdata.varidx['jy']]), ')')
print( 'BATS jz real mean (std):', np.mean(batsdata2[:,batsdata.varidx['jz']]), ' (', np.std(batsdata2[:,batsdata.varidx['jz']]), ')')
print('')

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].hist(batsdata.data_arr[:,batsdata.varidx['jx']], bins=BINS, label=r'$j_x$')
ax[1].hist(batsdata.data_arr[:,batsdata.varidx['jy']], bins=BINS, label=r'$j_y$')
ax[2].hist(batsdata.data_arr[:,batsdata.varidx['jz']], bins=BINS, label=r'$j_z$')

ax[0].set_xlabel(r'Line $j_x$')
ax[1].set_xlabel(r'Line $j_y$')
ax[2].set_xlabel(r'Line $j_z$')

# ax[0].set_xlim([-MINMAX,MINMAX])
# ax[1].set_xlim([-MINMAX,MINMAX])
# ax[2].set_xlim([-MINMAX,MINMAX])

print( 'BATS jx line mean (std):', np.mean(batsdata.data_arr[:,batsdata.varidx['jx']]), ' (', np.std(batsdata.data_arr[:,batsdata.varidx['jx']]), ')')
print( 'BATS jy line mean (std):', np.mean(batsdata.data_arr[:,batsdata.varidx['jy']]), ' (', np.std(batsdata.data_arr[:,batsdata.varidx['jy']]), ')')
print( 'BATS jz line mean (std):', np.mean(batsdata.data_arr[:,batsdata.varidx['jz']]), ' (', np.std(batsdata.data_arr[:,batsdata.varidx['jz']]), ')')
print('')

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].hist(oggcmdata2[:,oggcmdata.varidx['jx']], bins=BINS, label=r'$j_x$')
ax[1].hist(oggcmdata2[:,oggcmdata.varidx['jy']], bins=BINS, label=r'$j_y$')
ax[2].hist(oggcmdata2[:,oggcmdata.varidx['jz']], bins=BINS, label=r'$j_z$')

ax[0].set_xlabel(r'Real $j_x$')
ax[1].set_xlabel(r'Real $j_y$')
ax[2].set_xlabel(r'Real $j_z$')

# ax[0].set_xlim([-MINMAX,MINMAX])
# ax[1].set_xlim([-MINMAX,MINMAX])
# ax[2].set_xlim([-MINMAX,MINMAX])

print( 'Open jx real mean (std):', np.mean(oggcmdata2[:,oggcmdata.varidx['jx']]), ' (', np.std(oggcmdata2[:,oggcmdata.varidx['jx']]), ')')
print( 'Open jy real mean (std):', np.mean(oggcmdata2[:,oggcmdata.varidx['jy']]), ' (', np.std(oggcmdata2[:,oggcmdata.varidx['jy']]), ')')
print( 'Open jz real mean (std):', np.mean(oggcmdata2[:,oggcmdata.varidx['jz']]), ' (', np.std(oggcmdata2[:,oggcmdata.varidx['jz']]), ')')
print('')

fig, ax = plt.subplots(nrows=1, ncols=3)

ax[0].hist(oggcmdata.data_arr[:,oggcmdata.varidx['jx']], bins=BINS, label=r'$j_x$')
ax[1].hist(oggcmdata.data_arr[:,oggcmdata.varidx['jy']], bins=BINS, label=r'$j_y$')
ax[2].hist(oggcmdata.data_arr[:,oggcmdata.varidx['jz']], bins=BINS, label=r'$j_z$')

ax[0].set_xlabel(r'Line $j_x$')
ax[1].set_xlabel(r'Line $j_y$')
ax[2].set_xlabel(r'Line $j_z$')

# ax[0].set_xlim([-MINMAX,MINMAX])
# ax[1].set_xlim([-MINMAX,MINMAX])
# ax[2].set_xlim([-MINMAX,MINMAX])

print( 'Open jx line mean (std):', np.mean(oggcmdata.data_arr[:,oggcmdata.varidx['jx']]), ' (', np.std(oggcmdata.data_arr[:,oggcmdata.varidx['jx']]), ')')
print( 'Open jy line mean (std):', np.mean(oggcmdata.data_arr[:,oggcmdata.varidx['jy']]), ' (', np.std(oggcmdata.data_arr[:,oggcmdata.varidx['jy']]), ')')
print( 'Open jz line mean (std):', np.mean(oggcmdata.data_arr[:,oggcmdata.varidx['jz']]), ' (', np.std(oggcmdata.data_arr[:,oggcmdata.varidx['jz']]), ')')
print('')


# pltname = 'tot-Bned-Test-' + point
# fig.savefig( os.path.join( dtinfo['dir_plots'], 'BnedSurfInt_line_current', pltname + '.png' ) )
# plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.pdf' ) )
# plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.eps' ) )
# plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.jpg' ) )
# plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.tif' ) )
# plt.savefig( os.path.join( info['dir_plots'], 'BnedSurfInt', pltname + '.svg' ) )

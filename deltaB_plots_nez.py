#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: Dean Thomas
"""
################################################################################
################################################################################
# Derived from deltaB_plots.py. Most signficant difference is rather than
# plotting B in x,y,z components, its in north-east-zenith coordinates.
#
# Also refactored plotting code to use plot_2x4() routine below.  Reduced number
# of lines of code by 25%
#
# matplotlib was slow, so converted plotting to VTK
################################################################################
################################################################################

# Info on divB_simple1 runs available at:
#
# https://ccmc.gsfc.nasa.gov/results/viewrun.php?domain=GM&runnumber=Brian_Curtis_042213_7
#
# https://ccmc.gsfc.nasa.gov/RoR_WWW/GM/SWMF/2013/Brian_Curtis_042213_7/Brian_Curtis_042213_7_sw.gif
#

import logging
from copy import deepcopy
# import swmfio
from os import makedirs
from os.path import exists
import pandas as pd
import numpy as np
# from collections import namedtuple

from deltaB.plotting import plotargs, plotargs_multiy, create_directory, \
    plot_NxM, plot_NxM_multiy, pointcloud
from deltaB.BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_cumulative_sum_dataframe, create_jrtp_cdf_dataframes

COLABA = True

# origin and target define where input data and output plots are stored
if COLABA:
    origin = '/Volumes/Physics HD v2/runs/DIPTSUR2/GM/IO2/'
    target = '/Volumes/Physics HD v2/runs/DIPTSUR2/plots/'
else:
    origin = '/Volumes/Physics HD v2/divB_simple1/GM/'
    target = '/Volumes/Physics HD v2/divB_simple1/plots/'

# rCurrents define range from earth center below which results are not valid
# measured in Re units
if COLABA:
    rCurrents = 1.8
else:
    rCurrents = 3

# Range of values seen in each variable, used to plot graphs
if COLABA:
    rlog_limits = [1, 1000]
    r_limits = [0, 300]
    rho_limits = [10**-2, 10**4]
    p_limits = [10**-5, 10**3]
    jMag_limits = [10**-11, 10**1]
    j_limits = [-1, 1]
    jcdf_limits = [-0.1, 0.1]
    uMag_limits = [10**-3, 10**4]
    u_limits = [-1100, 1100]
    dBNorm_limits = [10**-15, 10**-1]

    dBx_sum_limits = [-1500, 1500]
    dBy_sum_limits = [-1500, 1500]
    dBz_sum_limits = [-1500, 1500]
    dBp_sum_limits = [-1500, 1500]
    dB_sum_limits = [0, 1500]
    dB_sum_limits2 = [-1200,200]

    plot3d_limits = [-10, 10]
    xyz_limits = [-300, 300]
    xyz_limits_small = [-20, 20]
    
    time_limits = [4,16]
    
    vmin = 0.02
    vmax = 0.5

else:
    rlog_limits = [1, 1000]
    r_limits = [0, 300]
    rho_limits = [10**-2, 10**2]
    p_limits = [10**-5, 10**2]
    jMag_limits = [10**-11, 10**0]
    j_limits = [-0.3, 0.3]
    jcdf_limits = [-0.1, 0.1]
    uMag_limits = [10**-3, 10**4]
    u_limits = [-1100, 1100]
    dBNorm_limits = [10**-15, 10**-1]
    
    dBx_sum_limits = [-0.4, 0.4]
    dBy_sum_limits = [-0.4, 0.4]
    dBz_sum_limits = [-50, 50]
    dBp_sum_limits = [-50, 50]
    dB_sum_limits = [0, 50]
    dB_sum_limits2 = [0, 50]
    
    plot3d_limits = [-10, 10]
    xyz_limits = [-300, 300]
    xyz_limits_small = [-20, 20]
    
    vmin = 0.007
    vmax = 0.30


# Range of values for cuts
if COLABA:
    cut1_jrmin = 0.02
    cut2_jphimin = 0.02
    cut2_rmin = 5 # 4 originally
    cut3_jphimin = 0.02
else:
    cut1_jrmin = 0.007
    cut1_y = 4
    cut2_jphimin = 0.007
    cut2_jphimax = 0.03
    cut3_jphimin = 0.007
    cut3_z = 2

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

##############################################################################
##############################################################################
# nez from weigel
# https://github.com/rweigel/magnetovis/blob/newapi2/magnetovis/Sources/BATSRUS_dB_demo.py#L45
##############################################################################
##############################################################################

def nez(time, pos, csys):
  """Unit vectors in geographic north, east, and zenith dirs"""

  from hxform import hxform as hx

  # Geographic z axis in csys
  Z = hx.transform(np.array([0, 0, 1]), time, 'GEO', csys, lib='cxform')

  # zenith direction ("up")
  z_geo = pos/np.linalg.norm(pos)

  e_geo = np.cross(Z, z_geo)
  e_geo = e_geo/np.linalg.norm(e_geo)

  # n_geo = np.cross(z_geo, e_geo)
  n_geo = np.cross(z_geo, e_geo)
  n_geo = n_geo/np.linalg.norm(n_geo)

  # print(f"Unit vectors for Geographic N, E, and Z in {csys}:")
  # print(f"North: {n_geo}")
  # print(f"East:  {e_geo}")
  # print(f"Z:     {z_geo}")

  return n_geo, e_geo, z_geo

def date_time(file):
    """Pull date and time from file basename

    Inputs:
        file = basename of file.  
    Outputs:
        month, day, year, hour, minute, second 
     """
    words = file.split('-')

    date = int(words[0].split('e')[1])
    y = date//10000
    n = (date % 10000) // 100
    d = date % 100

    # Convert time to a integers
    t = int(words[1])
    h = t//10000
    m = (t % 10000) // 100
    s = t % 100

    # logging.info(f'Time: {date} {t} Year: {y} Month: {n} Day: {d} Hours: {h} Minutes: {m} Seconds: {s}')

    return y, n, d, h, m, s

def get_files(orgdir=origin, base='3d__*'):
    """Create a list of files that we will process.  Look in the basedir directory,
    and get list of file basenames.

    Inputs:
        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out'
        orgdir = path to directory containing input files
    Outputs:
        l = list of file basenames.
    """
    import os
    import glob

    # Create a list of files that we will process
    # Look in the basedir directory.  Get list of file basenames

    # In this version, we find all of the base + '.out' files
    # and retrieve their basenames
    os.chdir(orgdir)

    l1 = glob.glob(base + '.out')

    # Strip off extension
    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    # Colaba incliudes 697 files, reduce the number by
    # accepting those only every 15 minutes
    if COLABA: 
        l2 = deepcopy(l1) 
        for i in range(len(l2)):
            y,n,d,h,m,s = date_time(l2[i])
            if( m % 15 != 0 ):
                l1.remove(l2[i])

    l1.sort()

    return l1

def get_files_unconverted(tgtsubdir='png-dBmagNorm-uMag-night',
                          orgdir=origin, tgtdir=target, base='3d__*'):
    """Create a list of files that we will process.  This routine is used when
    some files have been process and others have not, e.g., the program crashed.
    Since the output files of other routines use the same basenames as the output
    files, we compare the files in the input directory (orgdir) to those in the
    output directory (tgtdir).  From this, we create a list of unprocessed files.

    Inputs:
        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out'
        orgdir = path to directory containing input files
        tgtdir = path to directory containing output files
        tgtsubdir = the tgtdir contains multiple subdirectories containing output
            files from various routines.  tgtdir + tgtsubdir is the name of the
            directory with the output files that we will compare
    Outputs:
        l = list of file basenames
    """
    import os
    import glob

    # In this routine we compare the list of .out input files and .png files
    # to determine what has already been processed.  Look at all *.out
    # files and remove from the list (l1) all of the files that already have
    # been converted to .png files.  The unremoved files are unconverted files.

    os.chdir(orgdir)
    l1 = glob.glob(base + '.out')

    # Look at the png files in directory
    if not exists(tgtdir + tgtsubdir):
        makedirs(tgtdir + tgtsubdir)
    os.chdir(tgtdir + tgtsubdir)
    l2 = glob.glob(base + '.png')

    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    for i in range(len(l2)):
        l2[i] = (l2[i].split('.'))[0]

    for i in l2:
        l1.remove(i)

    # Colaba incliudes 697 files, reduce the number by
    # accepting those only every 15 minutes
    if COLABA: 
        l3 = deepcopy(l1) 
        for i in range(len(l3)):
            y,n,d,h,m,s = date_time(l3[i])
            if( m % 15 != 0 ):
                l1.remove(l3[i])

    l1.sort()

    return l1

def plot_db_Norm_r(df, title, base):
    """Plot components and magnitude of dB in each cell versus radius r.
    In this procedure, the dB values are normalized by cell volume

    Inputs:
        df = dataframe with BATSRUS data and calculated variables
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
     
    plots = [None] * 4
    
    plots[0] = plotargs(df, 'r', 'dBxNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_x |$ (Norm Cell Vol)', 
                        r_limits, dBNorm_limits, title)
    plots[1] = plotargs(df, 'r', 'dByNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_y |$ (Norm Cell Vol)', 
                        r_limits, dBNorm_limits, title)
    plots[2] = plotargs(df, 'r', 'dBzNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_z |$ (Norm Cell Vol)', 
                        r_limits, dBNorm_limits, title)
    plots[3] = plotargs(df, 'r', 'dBmagNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B |$ (Norm Cell Vol)', 
                        r_limits, dBNorm_limits, title)

    plot_NxM(target, base, 'png-dBNorm-r', plots, cols=4, rows=1 )
    
    return

def plot_dBnorm_various_day_night(df_day, df_night, title, base):
    """Plot dBmagNorm vs rho, p, magnitude of j, and magnitude of u in each cell.  

    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'rho', 'dBmagNorm', True, True, 
                        r'$\rho$', r'$| \delta B |$ (Norm Cell Vol)', 
                        rho_limits, dBNorm_limits, 'Day ' + title)
    plots[1] = plotargs(df_day, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                        p_limits, dBNorm_limits, 'Day ' + title)
    plots[2] = plotargs(df_day, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        jMag_limits, dBNorm_limits, 'Day ' + title)
    plots[3] = plotargs(df_day, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        uMag_limits, dBNorm_limits, 'Day ' + title)
    plots[4] = plotargs(df_night, 'rho', 'dBmagNorm', True, True, 
                        r'$\rho$', r'$| \delta B |$ (Norm Cell Vol)', 
                        rho_limits, dBNorm_limits, 'Night ' + title)
    plots[5] = plotargs(df_night, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                        p_limits, dBNorm_limits, 'Night ' + title)
    plots[6] = plotargs(df_night, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        jMag_limits, dBNorm_limits, 'Night ' + title)
    plots[7] = plotargs(df_night, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        uMag_limits, dBNorm_limits, 'Night ' + title)

    plot_NxM(target, base, 'png-dBNorm-various-day-night', plots )
    
    return

def plot_sum_dB(df_r, title, base):
    """Plot various forms of the cumulative sum of dB in each cell versus 
        range r.  To generate the cumulative sum, we order the cells in terms of
        range r from the earth's center.  We start with a small sphere and 
        vector sum all of the dB contributions inside the sphere.  Expand the
        sphere slightly and resum.  Repeat until all cells are in the sum.

    Inputs:
        df_r = dataframe containing cumulative sums ordered from small r to 
            large r
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
    plots = [None] * 4
    
    plots[0] = plotargs(df_r, 'r', 'dBxSum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_x $', 
                        rlog_limits, dBx_sum_limits, title)
    plots[1] = plotargs(df_r, 'r', 'dBySum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_y $', 
                        rlog_limits, dBy_sum_limits, title)
    plots[2] = plotargs(df_r, 'r', 'dBzSum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_z $', 
                        rlog_limits, dBz_sum_limits, title)
    plots[3] = plotargs(df_r, 'r', 'dBSumMag', True, False, 
                        r'$r/R_E$', r'$| \Sigma_r \delta B |$', 
                        rlog_limits, dB_sum_limits, title)
    
    plot_NxM(target, base, 'png-sum-dB-r', plots, cols=4, rows=1 )

    return

def plot_cumulative_B_para_perp(df_r, title, base):
    """Plot various forms of the cumulative sum of dB in each cell versus 
        range r.  These plots examine the contributions of the currents
        parallel and perpendicular to the magnetic field.  
        To generate the cumulative sum, we order the cells in terms of
        range r from the earth's center.  We start with a small sphere and 
        vector sum all of the dB contributions inside the sphere.  Expand the
        sphere slightly and resum.  Repeat until all cells are in the sum.

    Inputs:
        df_r = dataframe containing cumulative sums ordered from small r to 
            large r
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs_multiy(df_r, 'r', 
                        ['dBparallelxSum'], 
                        True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_x (j_{\parallel})$',
                        [r'$\parallel$'], 
                        rlog_limits, dBx_sum_limits, r'$\parallel$ ' + title)

    plots[4] = plotargs_multiy(df_r, 'r', 
                        ['dBperpendicularxSum', 'dBperpendicularphixSum', 'dBperpendicularphiresxSum'], 
                        True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_x (j_{\perp})$',
                        [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                        rlog_limits, dBx_sum_limits, r'$\perp$ ' + title)

    plots[1] = plotargs_multiy(df_r, 'r', 
                        ['dBparallelySum'], 
                        True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\parallel})$',
                        [r'$\parallel$'], 
                        rlog_limits, dBy_sum_limits, r'$\parallel$ ' + title)

    plots[5] = plotargs_multiy(df_r, 'r', 
                        ['dBperpendicularySum', 'dBperpendicularphiySum', 'dBperpendicularphiresySum'], 
                        True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\perp})$',
                        [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                        rlog_limits, dBy_sum_limits, r'$\perp$ ' + title)

    plots[2] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelzSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\parallel})$',
                     [r'$\parallel$'], 
                     rlog_limits, dBz_sum_limits, r'$\parallel$ ' + title)
    
    plots[6] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularzSum', 'dBperpendicularphizSum', 'dBperpendicularphireszSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\perp})$',
                     [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                     rlog_limits, dBz_sum_limits, r'$\perp$ ' + title)

    plots[3] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\parallel)|$',
                     [r'$\parallel$'], 
                     rlog_limits, dBz_sum_limits, r'$\parallel$ ' + title)
    
    plots[7] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\perp)|$',
                     [r'$\perp$'], 
                     rlog_limits, dBz_sum_limits, r'$\perp$ ' + title)
        
    plot_NxM_multiy(target, base, 'png-sum-dB-para-perp-comp-r', plots, plottype = 'line')

    return

def plot_rho_p_jMag_uMag_day_night(df_day, df_night, title, base):
    """Plot rho, p, magnitude of j, and magnitude of u in each cell versus 
        radius r.  

    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'rho', True, True, 
                        r'$r$', r'$\rho$', 
                        rlog_limits, rho_limits, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'p', True, True, 
                        r'$r$', r'$p$', 
                        rlog_limits, p_limits, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$', 
                        rlog_limits, jMag_limits, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$', 
                        rlog_limits, uMag_limits, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'rho', True, True, 
                        r'$r$', r'$\rho$', 
                        rlog_limits, rho_limits, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'p', True, True, 
                        r'$r$', r'$p$',  
                        rlog_limits, p_limits, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$',  
                        rlog_limits, jMag_limits, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$',  
                        rlog_limits, uMag_limits, 'Night ' + title)

    plot_NxM(target, base, 'png-various-r-day-night', plots )
    
    return

def plot_jx_jy_jz_day_night(df_day, df_night, title, base):
    """Plot jx, jy, jz  in each cell versus radius r.  

    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'jx', True, False, 
                        r'$ r/R_E $', r'$j_x$',
                        rlog_limits, j_limits, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$', 
                        rlog_limits, j_limits, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$', 
                        rlog_limits, j_limits, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$', 
                        rlog_limits, j_limits, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'jx', True, False, 
                        r'$ r/R_E $', r'$j_x$', 
                        rlog_limits, j_limits, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$',  
                        rlog_limits, j_limits, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$',  
                        rlog_limits, j_limits, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$',  
                        rlog_limits, j_limits, 'Night ' + title)

    plot_NxM(target, base, 'png-jxyz-r-day-night', plots )

    return

def plot_ux_uy_uz_day_night(df_day, df_night, title, base):
    """Plot ux, uy, uz  in each cell versus radius r.  

    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'ux', True, False, 
                        r'$ r/R_E $', r'$u_x$',
                        rlog_limits, u_limits, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$', 
                        rlog_limits, u_limits, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$', 
                        rlog_limits, u_limits, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$', 
                        rlog_limits, u_limits, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'ux', True, False, 
                        r'$ r/R_E $', r'$u_x$', 
                        rlog_limits, u_limits, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$',  
                        rlog_limits, u_limits, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$',  
                        rlog_limits, u_limits, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$',  
                        rlog_limits, u_limits, 'Night ' + title)

    plot_NxM(target, base, 'png-uxyz-r-day-night', plots )

    return

def plot_jr_jt_jp_vs_x(df, title, base, coord='x', cut=''):
    """Plot jr, jtheta, jphi  in each cell versus x.  

    Inputs:
        df = dataframe containing jr, jtheta, jphi and x
        title = title for plots
        base = basename of file where plot will be stored
        coord = which coordinate is on the x axis - x, y, or z
        cut = which cut was made, used in plot filenames
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df, coord, 'jr', False, False, 
                    r'$' + coord + '/R_E$', r'$j_r$',
                    xyz_limits, j_limits, title)
    plots[1] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$', 
                    xyz_limits, j_limits, title)
    plots[2] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$', 
                    xyz_limits, j_limits, title)
    plots[3] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                    xyz_limits, j_limits, title)
    plots[4] = plotargs(df, coord, 'jr', False, False, 
                    r'$' + coord + '/R_E$', r'$j_r$', 
                    xyz_limits_small, j_limits, title)
    plots[5] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$',  
                    xyz_limits_small, j_limits, title)
    plots[6] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$',  
                    xyz_limits_small, j_limits, title)
    plots[7] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                    xyz_limits_small, j_limits, title)
    
    plot_NxM(target, base, 'png-jrtp-'+cut+coord, plots )

    return

def plot_jp_jp_vs_x(df, title, base, coord='x', cut=''):
    """Plot jparallel and jperpendicular to the B field in each cell versus x.  

    Inputs:
        df = dataframe containing jparallel, jperpendicular, and x
        title = title for plots
        base = basename of file where plot will be stored
        coord = which coordinate is on the x axis - x, y, or z
        cut = which cut was made, used in plot filenames
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df, coord, 'jparallelMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\parallel$',
                    xyz_limits, j_limits, title)
    plots[1] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$', 
                    xyz_limits, j_limits, title)
    plots[2] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                    xyz_limits, j_limits, title)
    plots[4] = plotargs(df, coord, 'jparallelMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\parallel$', 
                    xyz_limits_small, j_limits, title)
    plots[5] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$',  
                    xyz_limits_small, j_limits, title)
    plots[6] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                    xyz_limits_small, j_limits, title)
    
    plot_NxM(target, base, 'png-jpp-'+cut+coord, plots, cols=3, rows=2 )

    return

def plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base):
    """Plot jr, jtheta, jphi CDFs  

    Inputs:
        df_jr, df_jtheta, df_jphi = dataframe CDF data
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    plots = [None] * 3
     
    plots[0] = plotargs(df_jr, 'jr', 'cdfIndex', False, False, 
                    r'$j_r$', r'$CDF$',
                    jcdf_limits, [0,1], title)
    plots[1] = plotargs(df_jtheta, 'jtheta', 'cdfIndex', False, False, 
                    r'$j_\theta$', r'$CDF$',
                    jcdf_limits, [0,1], title)
    plots[2] = plotargs(df_jphi, 'jphi', 'cdfIndex', False, False, 
                    r'$j_\phi$', r'$CDF$',
                    jcdf_limits, [0,1], title)
    
    plot_NxM(target, base, 'png-jrtp-cdf', plots, cols=3, rows=1 )

    return

def process_data(X, Y, Z, base, dirpath=origin):
    """Process data in BATSRUS file to create dataframe with calculated quantities.

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
        title = title to use in plots, which is derived from base (file basename)
        batsrus = BATSRUS data read by swmfio 
    """

    df, title = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin, rCurrents=rCurrents)

    logging.info('Creating cumulative sum dB dataframe...')

    df_r = create_cumulative_sum_dataframe(df)

    # logging.info('Creating dayside/nightside dataframe...')
    # df_day = df[df['x'] >= 0]
    # df_night = df[df['x'] < 0]

    # Do plots...

    # logging.info('Creating dB (Norm) vs r plots...')
    # plot_db_Norm_r( df, title, base )
    
    # logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    # plot_dBnorm_various_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating cumulative sum B vs r plots...')
    # plot_sum_dB( df_r, title, base )

    logging.info('Creating cumulative sum B parallel/perpendicular vs r plots...')
    plot_cumulative_B_para_perp(df_r, title, base)

    # logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    # plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base )

    # logging.info('Creating day /night jx, jy, jz vs r plots...')
    # plot_jx_jy_jz_day_night( df_day, df_night, title, base )

    # logging.info('Creating day/night ux, uy, uz vs r plots...')
    # plot_ux_uy_uz_day_night( df_day, df_night, title, base )

    # logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'x')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'y')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x( df, title, base, coord = 'x')
    # plot_jp_jp_vs_x( df, title, base, coord = 'y')
    # plot_jp_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jrtp CDFs...')
    # df_jr, df_jtheta, df_jphi = create_jrtp_cdf_dataframes(df)
    # plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base)

    return

def perform_cuts(df1, title1, cut_selected):
    """perform selected cuts on BATSRUS dataframe, df1.

    Inputs:
        df1 = BATSRUS dataframe on which to make cuts
        title2 = base title for plots, will be modified based on cuts
        cut_selected = which cut to make
    Outputs:
        df2 = dataframe with cuts applied
        title2 = title to be used in plots
        cutname = string signifying cut
    """

    assert(0 < cut_selected < 4)

    df_tmp = deepcopy(df1)

    # Cut asymmetric jr vs y lobes, always make this cut
    df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
    cutname = r'jr-'
    title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected > 1):
        # Cut jphi vs y blob
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
        cutname = 'jphifar-' + cutname
        title2 = r'$j_\phi$ Peaks (far) ' + title2

    if(cut_selected > 2):
        # Cut jphi vs z blob
        df2 = df2.drop(df2[df2['jphi'].abs() > cut3_jphimin].index)
        cutname = 'jphinear-' + cutname
        title2 = r'$j_\phi$ (near) ' + title2

    return df2, title2, cutname

def perform_not_cuts(df1, title1, cut_selected):
    """perform selected cuts on BATSRUS dataframe, df1. Creates the opposite result
    of perform_cuts

    Inputs:
        df1 = BATSRUS dataframe on which to make cuts
        title2 = base title for plots, will be modified based on cuts
        cut_selected = which cut to make
    Outputs:
        df2 = dataframe with cuts applied
        title2 = title to be used in plots
        cutname = string signifying cut
    """

    assert(0 < cut_selected < 5)

    df_tmp = deepcopy(df1)

    # Cut asymmetric jr vs y lobes, always make this cut
    if(cut_selected == 1):
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() <= cut1_jrmin].index)
        cutname = 'jr-'
        title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected == 2):
        # Cut jphi vs y blob
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'].abs() <= cut2_jphimin, df2['r'] <= cut2_rmin)].index)
        cutname = 'jphifar-'
        title2 = r'$j_\phi$ Peaks (far) ' + title1

    if(cut_selected == 3):
        # Cut jphi vs z blob
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() <= cut3_jphimin].index)
        cutname = 'jphinear-'
        title2 = r'$j_\phi$ (near) ' + title1

    if(cut_selected == 4):
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() > cut3_jphimin].index)
        cutname = 'residual-'
        title2 = r'Residual ' + title1

    return df2, title2, cutname

def process_data_with_cuts(X, Y, Z, base, dirpath=origin, cut_selected=1):
    """Process data in BATSRUS file to create dataframe with calculated quantities.

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
        title = title to use in plots, which is derived from base (file basename)
        batsrus = BATSRUS data read by swmfio 
    """

    # Read BASTRUS file
    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin, rCurrents=rCurrents)

    # Perform cuts on BATSRUS data
    df2, title2, cutname = perform_cuts(df1, title1, cut_selected=cut_selected)

    # Do plots...

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='x', cut=cutname)
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='y', cut=cutname)
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='z', cut=cutname)

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x(df2, title2, base, coord='x', cut=cutname)
    # plot_jp_jp_vs_x(df2, title2, base, coord='y', cut=cutname)
    # plot_jp_jp_vs_x(df2, title2, base, coord='z', cut=cutname)

    return

def process_3d_cut_plots(X, Y, Z, base, dirpath=origin):
    """Process data in BATSRUS file to create 3D plots of points in cuts

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        None - other than the saved plot file
    """

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin, rCurrents=rCurrents)

    logging.info('Creating dataframes with extracted cuts...')

    #################################
    #################################
    # Cut asymmetric jr vs y lobes
    df2, title2, cutname2 = perform_not_cuts(df1, title1, cut_selected=1)
    #################################
    #################################
    # Cut jphi vs y blob
    df3, title3, cutname3 = perform_not_cuts(df1, title1, cut_selected=2)
    #################################
    #################################
    # Cut jphi vs z blob
    df4, title4, cutname4 = perform_not_cuts(df1, title1, cut_selected=3)
    #################################
    #################################

    logging.info('Plotting 3D extracted cuts...')

    xyz = ['x', 'y', 'z']
    colorvars = ['jr', 'jtheta', 'jphi', 'jparallelMag', 'jperpendicularMag']
    
    logging.info(f'Saving {base} 3D cut plots')

    cuts2 = pointcloud( df2, xyz, colorvars )
    cuts2.convert_to_vtk()
    # cuts2.display_vtk()
    cuts2.write_vtk_to_file( target, base, 'vtk-3d-cut1' )
    
    cuts3 = pointcloud( df3, xyz, colorvars )
    cuts3.convert_to_vtk()
    # cuts3.display_vtk()
    cuts3.write_vtk_to_file( target, base, 'vtk-3d-cut2' )
    
    cuts4 = pointcloud( df4, xyz, colorvars )
    cuts4.convert_to_vtk()
    # cuts4.display_vtk()
    cuts4.write_vtk_to_file( target, base, 'vtk-3d-cut3' )
    
    return

def process_sum_db_with_cuts(X, Y, Z, base, dirpath=origin):
    """Process data in BATSRUS file to create dataframe with calculated quantities,
    but in this case we perform some cuts on the data to isolate high current
    regions.  This cut data is used to determine the fraction of the total B
    field due to each set of currents

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
     Outputs:
        df1 = cumulative sum for original (all) data in north-east-zenith
        df1 - df2 = contribution due to points in asym. jr cut in nez
        df2 - df3 = contribution due to points in y jphi cut in nez
        df3 - df4 = contribution due to points in z jphi cut in nez
    """

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin, rCurrents=rCurrents)

    logging.info('Creating dataframes with extracted cuts...')

    #################################
    #################################
    # Cut asymmetric jr vs y lobes
    df2, title2, cutname2 = perform_cuts(df1, title1, cut_selected=1)
    #################################
    #################################
    # Cut jphi vs y blob
    df3, title3, cutname3 = perform_cuts(df1, title1, cut_selected=2)
    #################################
    #################################
    # Cut jphi vs z blob
    df4, title4, cutname4 = perform_cuts(df1, title1, cut_selected=3)
    #################################
    #################################

    logging.info('Calculate cumulative sums for dataframes for extracted cuts...')

    df1 = create_cumulative_sum_dataframe(df1)
    df2 = create_cumulative_sum_dataframe(df2)
    df3 = create_cumulative_sum_dataframe(df3)
    df4 = create_cumulative_sum_dataframe(df4)
    
    y,n,d,h,m,s = date_time(base)
    # n_geo, e_geo, z_geo = nez((y,n,d,h,m,s), (X,Y,Z), 'GSM')
    n_geo = (0,0,1)
    
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

    dBNSum2, dBparallelNSum2, dBperpendicularNSum2, dBperpendicularphiNSum2, \
        dBperpendicularphiresNSum2 = north_comp( df2, n_geo )

    dBNSum3, dBparallelNSum3, dBperpendicularNSum3, dBperpendicularphiNSum3, \
        dBperpendicularphiresNSum3 = north_comp( df3, n_geo )

    dBNSum4, dBparallelNSum4, dBperpendicularNSum4, dBperpendicularphiNSum4, \
        dBperpendicularphiresNSum4 = north_comp( df4, n_geo )

    return dBNSum1, \
        dBparallelNSum1, \
        dBperpendicularNSum1, \
        dBperpendicularphiNSum1, \
        dBperpendicularphiresNSum1, \
        dBNSum1 - dBNSum2, \
        dBparallelNSum1 - dBparallelNSum2, \
        dBperpendicularNSum1 - dBperpendicularNSum2, \
        dBperpendicularphiNSum1 - dBperpendicularphiNSum2, \
        dBperpendicularphiresNSum1 - dBperpendicularphiresNSum2, \
        dBNSum2 - dBNSum3, \
        dBparallelNSum2 - dBparallelNSum3, \
        dBperpendicularNSum2 - dBperpendicularNSum3, \
        dBperpendicularphiNSum2 - dBperpendicularphiNSum3, \
        dBperpendicularphiresNSum2 - dBperpendicularphiresNSum3, \
        dBNSum3 - dBNSum4, \
        dBparallelNSum3 - dBparallelNSum4, \
        dBperpendicularNSum3 - dBperpendicularNSum4, \
        dBperpendicularphiNSum3 - dBperpendicularphiNSum4, \
        dBperpendicularphiresNSum3 - dBperpendicularphiresNSum4, \
        dBNSum4, \
        dBparallelNSum4, \
        dBperpendicularNSum4, \
        dBperpendicularphiNSum4, \
        dBperpendicularphiresNSum4

def process_sum_db(X, Y, Z, base, dirpath=origin):
    """Process data in BATSRUS file to create dataframe with calculated quantities

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
     Outputs:
        df1 = cumulative sum for original (all) data in north-east-zenith
    """

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin, rCurrents=rCurrents)

    logging.info('Calculate cumulative sums for dataframes for extracted cuts...')

    df1 = create_cumulative_sum_dataframe(df1)
    
    y,n,d,h,m,s = date_time(base)
    # n_geo, e_geo, z_geo = nez((y,n,d,h,m,s), (X,Y,Z), 'GSM')
    n_geo = (0,0,1)
    
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

def loop_sum_db_thru_cuts(X, Y, Z, files):
    """Loop thru data in BATSRUS files to create plots showing the effects of
    various cuts on the data.  See process_data_with_cuts for the specific cuts 
    made

    Inputs:
        files = list of files to be processed, each entry is the basename of 
            a BATSRUS file
    Outputs:
        None - other than the plots generated
    """
    n = len(files)
    # n = 4

    b_original = [None] * n
    b_original_parallel = [None] * n
    b_original_perp = [None] * n
    b_original_perpphi = [None] * n
    b_original_perpphires = [None] * n
    b_asym_jr = [None] * n
    b_asym_jr_parallel = [None] * n
    b_asym_jr_perp = [None] * n
    b_asym_jr_perpphi = [None] * n
    b_asym_jr_perpphires = [None] * n
    b_y_jphi = [None] * n
    b_y_jphi_parallel = [None] * n
    b_y_jphi_perp = [None] * n
    b_y_jphi_perpphi = [None] * n
    b_y_jphi_perpphires = [None] * n
    b_z_jphi = [None] * n
    b_z_jphi_parallel = [None] * n
    b_z_jphi_perp = [None] * n
    b_z_jphi_perpphi = [None] * n
    b_z_jphi_perpphires = [None] * n
    b_residual = [None] * n
    b_residual_parallel = [None] * n
    b_residual_perp = [None] * n
    b_residual_perpphi = [None] * n
    b_residual_perpphires = [None] * n
    b_times = [None] * n
    b_index = [None] * n

    for i in range(n):        
        # Create the title that we'll use in the graphics
        words = files[i].split('-')

        # Convert time to a float
        t = int(words[1])
        h = t//10000
        m = (t % 10000) // 100
        logging.info(f'Time: {t} Hours: {h} Minutes: {m}')

        # Record time and index for plots
        b_times[i] = h + m/60
        b_index[i] = i

        # Get the values of various cuts on the data.  We want the sums for the
        # main components of the field - the complete field, parallel and perpendicular
        # (perpphi and perpphires are components of perpendicular)
        b_original[i], b_original_parallel[i], b_original_perp[i], \
            b_original_perpphi[i], b_original_perpphires[i], \
            b_asym_jr[i], b_asym_jr_parallel[i], b_asym_jr_perp[i], \
            b_asym_jr_perpphi[i], b_asym_jr_perpphires[i], \
            b_y_jphi[i], b_y_jphi_parallel[i], b_y_jphi_perp[i], \
            b_y_jphi_perpphi[i], b_y_jphi_perpphires[i], \
            b_z_jphi[i], b_z_jphi_parallel[i], b_z_jphi_perp[i], \
            b_z_jphi_perpphi[i], b_z_jphi_perpphires[i], \
            b_residual[i], b_residual_parallel[i], b_residual_perp[i], \
            b_residual_perpphi[i], b_residual_perpphires[i] = \
            process_sum_db_with_cuts(X, Y, Z, base=files[i])

    df = pd.DataFrame( { r'Total': b_original, 
                        r'Parallel': b_original_parallel, 
                        r'Perpendicular': b_original_perp, 
                        r'Perpendicular $\phi$': b_original_perpphi, 
                        r'Perpendicular Residual': b_original_perpphires,
                        r'Time (hr)': b_times,
                        r'$j_r$ Total': b_asym_jr, 
                        r'$j_r$ Parallel': b_asym_jr_parallel, 
                        r'$j_r$ Perpendicular': b_asym_jr_perp, 
                        r'$j_r$ Perpendicular $\phi$': b_asym_jr_perpphi, 
                        r'$j_r$ Perpendicular Residual': b_asym_jr_perpphires,
                        r'$y j_\phi$ Total': b_y_jphi, 
                        r'$y j_\phi$ Parallel': b_y_jphi_parallel, 
                        r'$y j_\phi$ Perpendicular': b_y_jphi_perp, 
                        r'$y j_\phi$ Perpendicular $\phi$': b_y_jphi_perpphi, 
                        r'$y j_\phi$ Perpendicular Residual': b_y_jphi_perpphires,
                        r'$z j_\phi$ Total': b_z_jphi, 
                        r'$z j_\phi$ Parallel': b_z_jphi_parallel, 
                        r'$z j_\phi$ Perpendicular': b_z_jphi_perp, 
                        r'$z j_\phi$ Perpendicular $\phi$': b_z_jphi_perpphi, 
                        r'$z j_\phi$ Perpendicular Residual': b_z_jphi_perpphires,
                        r'Residual Total': b_residual, 
                        r'Residual Parallel': b_residual_parallel, 
                        r'Residual Perpendicular': b_residual_perp, 
                        r'Residual Perpendicular $\phi$': b_residual_perpphi, 
                        r'Residual Perpendicular Residual': b_residual_perpphires })

    plots = [None] * 5
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['Total', r'Parallel', r'$Perpendicular $\phi$', r'Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'All')   
        
    plots[1] = plotargs_multiy(df, r'Time (hr)', 
                        ['$j_r$ Total', r'$j_r$ Parallel', r'$j_r$ Perpendicular $\phi$', r'$j_r$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['$j_r$ Total', r'$j_r$ Parallel', r'$j_r$ Perpendicular $\phi$', r'$j_r$ Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'$j_r$')   
        
    plots[2] = plotargs_multiy(df, r'Time (hr)', 
                        ['$y j_\phi$ Total', r'$y j_\phi$ Parallel', r'$y j_\phi$ Perpendicular $\phi$', r'$y j_\phi$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['$y j_\phi$ Total', r'$y j_\phi$ Parallel', r'$y j_\phi$ Perpendicular $\phi$', r'$y j_\phi$ Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'$y j_\phi$')   
        
    plots[3] = plotargs_multiy(df, r'Time (hr)', 
                        ['$z j_\phi$ Total', r'$z j_\phi$ Parallel', r'$z j_\phi$ Perpendicular $\phi$', r'$z j_\phi$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['$z j_\phi$ Total', r'$z j_\phi$ Parallel', r'$z j_\phi$ Perpendicular $\phi$', r'$z j_\phi$ Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'$z j_\phi$')   
        
    plots[4] = plotargs_multiy(df, r'Time (hr)', 
                        ['Residual Total', r'Residual Parallel', r'Residual Perpendicular $\phi$', r'Residual Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['Residual Total', r'Residual Parallel', r'Residual Perpendicular $\phi$', r'Residual Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'$z j_\phi$')   
        
    plot_NxM_multiy(target, 'para-perp-by-peak', 'parallel-perpendicular-composition', 
                    plots, cols=5, rows=1, plottype = 'line')

    return

def loop_sum_db(X, Y, Z, files):
    """Loop thru data in BATSRUS files to create plots showing the breakdown of
    parallel and perpendicular to B field components

    Inputs:
        files = list of files to be processed, each entry is the basename of 
            a BATSRUS file
    Outputs:
        None - other than the plots generated
    """
    n = len(files)
    # n = 4

    b_original = [None] * n
    b_original_parallel = [None] * n
    b_original_perp = [None] * n
    b_original_perpphi = [None] * n
    b_original_perpphires = [None] * n
    b_times = [None] * n
    b_index = [None] * n

    for i in range(n):        
        # Create the title that we'll use in the graphics
        words = files[i].split('-')

        # Convert time to a float
        t = int(words[1])
        h = t//10000
        m = (t % 10000) // 100
        logging.info(f'Time: {t} Hours: {h} Minutes: {m}')

        # Record time and index for plots
        b_times[i] = h + m/60
        b_index[i] = i

        # Get the values of various cuts on the data.  We want the sums for the
        # main components of the field - the complete field, parallel and perpendicular
        # (perpphi and perpphires are components of perpendicular)
        b_original[i], b_original_parallel[i], b_original_perp[i], \
            b_original_perpphi[i], b_original_perpphires[i] = \
            process_sum_db(X, Y, Z, base=files[i])

    b_fraction_parallel = [m/n for m, n in zip(b_original_parallel, b_original)]
    b_fraction_perpphi = [m/n for m, n in zip(b_original_perpphi, b_original)]
    b_fraction_perpphires = [m/n for m, n in zip(b_original_perpphires, b_original)]

    df = pd.DataFrame( { r'Total': b_original, 
                        r'Parallel': b_original_parallel, 
                        r'Perpendicular': b_original_perp, 
                        r'Perpendicular $\phi$': b_original_perpphi, 
                        r'Perpendicular Residual': b_original_perpphires,
                        r'Time (hr)': b_times,
                        r'Fraction Parallel': b_fraction_parallel, 
                        r'Fraction Perpendicular $\phi$': b_fraction_perpphi, 
                        r'Fraction Perpendicular Residual': b_fraction_perpphires } )

    plots = [None] * 5
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_z$ at (1,0,0)',
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        time_limits, dB_sum_limits2, r'Total')   
        
    plots[1] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Parallel', r'Fraction Perpendicular $\phi$', r'Fraction Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Fraction of Total $B_z$ at (1,0,0)',
                        [r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        time_limits, [-1,1], r'Fraction')
    
    plots[2] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Parallel'], 
                        False, False, 
                        r'Time (hr)',
                        r'Fraction of Total $B_z$ at (1,0,0)',
                        [r'Parallel'], 
                        time_limits, [-1,1], r'Fraction Parallel')
    
    plots[3] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Perpendicular $\phi$'], 
                        False, False, 
                        r'Time (hr)',
                        r'Fraction of Total $B_z$ at (1,0,0)',
                        [r'Perpendicular $\phi$'], 
                        time_limits, [-1,1], r'Fraction Perpendicular $\phi$')
    
    plots[4] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Fraction of Total $B_z$ at (1,0,0)',
                        [r'Perpendicular Residual'], 
                        time_limits, [-1,1], r'Fraction Perpendicular Residual')
    
    plot_NxM_multiy(target, 'para-perp', 'parallel-perpendicular-composition', 
                    plots, cols=5, rows=1, plottype = 'line')
    
    return

import sys, getopt

def main(argv):

    inputfile = ''

    opts, args = getopt.getopt(argv,"hi:")
    for opt, arg in opts:
      if opt == '-h':
          print ('test.py -i <inputfile>')
          sys.exit()
      elif opt in ("-i"):
          inputfile = arg

    X = 1
    Y = 0
    Z = 0

    # print("test: " + inputfile)
    process_data(X, Y, Z, inputfile)
    # process_data_with_cuts(X, Y, Z, inputfile, cut_selected = 3)
    # process_3d_cut_plots(X, Y, Z, inputfile)
    # process_3d_cut_plots(X, Y, Z, inputfile)

def main2(argv):
    if COLABA:
        files = get_files(base='3d__var_2_e*')
    else:
        files = get_files(base='3d__*')
        
    logging.info('Num. of files: ' + str(len(files)))

    X = 1
    Y = 0
    Z = 0

    loop_sum_db_thru_cuts(X, Y, Z, files)
    loop_sum_db(X, Y, Z, files)
    

if __name__ == "__main__":
   main2(sys.argv[1:])


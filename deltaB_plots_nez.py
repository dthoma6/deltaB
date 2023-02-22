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
#
# converted constants, eg., origin to caps ORIGIN
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
# from os import makedirs
from os.path import exists
import pandas as pd
import numpy as np
# from collections import namedtuple

from deltaB.plotting import plotargs, plotargs_multiy, \
    plot_NxM, plot_NxM_multiy, pointcloud
from deltaB.BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_cumulative_sum_dataframe, \
    create_jrtp_cdf_dataframes, \
    calc_gap_dB
from deltaB.util import ned, date_time, date_timeISO, get_files
from deltaB.util import create_directory

COLABA = True

# origin and target define where input data and output plots are stored
if COLABA:
    ORIGIN = '/Volumes/Physics HD v2/runs/DIPTSUR2/GM/IO2/'
    TARGET = '/Volumes/Physics HD v2/runs/DIPTSUR2/plots/'
else:
    ORIGIN = '/Volumes/Physics HD v2/divB_simple1/GM/'
    TARGET = '/Volumes/Physics HD v2/divB_simple1/plots/'

# rCurrents define range from earth center below which results are not valid
# measured in Re units
if COLABA:
    RCURRENTS = 1.8
else:
    RCURRENTS = 3

# Range of values seen in each variable, used to plot graphs
if COLABA:
    RLOG_LIMITS = [1, 1000]
    R_LIMITS = [0, 300]
    RHO_LIMITS = [10**-2, 10**4]
    P_LIMITS = [10**-5, 10**3]
    JMAG_LIMITS = [10**-11, 10**1]
    J_LIMITS = [-1, 1]
    JCDF_LIMITS = [-0.1, 0.1]
    UMAG_LIMITS = [10**-3, 10**4]
    U_LIMITS = [-1100, 1100]
    DBNORM_LIMITS = [10**-15, 10**-1]

    DBX_SUM_LIMITS = [-1500, 1500]
    DBY_SUM_LIMITS = [-1500, 1500]
    DBZ_SUM_LIMITS = [-1500, 1500]
    DBP_SUM_LIMITS = [-1500, 1500]
    DB_SUM_LIMITS = [0, 1500]
    DB_SUM_LIMITS2 = [-1200,400]

    PLOT3D_LIMITS = [-10, 10]
    XYZ_LIMITS = [-300, 300]
    XYZ_LIMITS_SMALL = [-20, 20]
    
    TIME_LIMITS = [4,16]
    FRAC_LIMITS = [-0.5,1.5]
    
    VMIN = 0.02
    VMAX = 0.5

else:
    RLOG_LIMITS = [1, 1000]
    R_LIMITS = [0, 300]
    RHO_LIMITS = [10**-2, 10**2]
    P_LIMITS = [10**-5, 10**2]
    JMAG_LIMITS = [10**-11, 10**0]
    J_LIMITS = [-0.3, 0.3]
    JCDF_LIMITS = [-0.1, 0.1]
    UMAG_LIMITS = [10**-3, 10**4]
    U_LIMITS = [-1100, 1100]
    DBNORM_LIMITS = [10**-15, 10**-1]
    
    DBX_SUM_LIMITS = [-0.4, 0.4]
    DBY_SUM_LIMITS = [-0.4, 0.4]
    DBZ_SUM_LIMITS = [-50, 50]
    DBP_SUM_LIMITS = [-50, 50]
    DB_SUM_LIMITS = [0, 50]
    DB_SUM_LIMITS2 = [0, 50]
    
    PLOT3D_LIMITS = [-10, 10]
    XYZ_LIMITS = [-300, 300]
    XYZ_LIMITS_SMALL = [-20, 20]
    
    VMIN = 0.007
    VMAX = 0.30


# Range of values for cuts
if COLABA:
    CUT1_JRMIN = 0.02
    CUT2_JPHIMIN = 0.02
    CUT2_RMIN = 5 # 4 ORIGINALLY
    CUT3_JPHIMIN = 0.02
else:
    CUT1_JRMIN = 0.007
    CUT2_JPHIMIN = 0.007
    CUT2_RMIN = 2
    CUT3_JPHIMIN = 0.007

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

#############################################################################
#############################################################################
# plot_... routines generate various plots of BATSRUS data
#############################################################################
#############################################################################

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
                        R_LIMITS, DBNORM_LIMITS, title)
    plots[1] = plotargs(df, 'r', 'dByNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_y |$ (Norm Cell Vol)', 
                        R_LIMITS, DBNORM_LIMITS, title)
    plots[2] = plotargs(df, 'r', 'dBzNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_z |$ (Norm Cell Vol)', 
                        R_LIMITS, DBNORM_LIMITS, title)
    plots[3] = plotargs(df, 'r', 'dBmagNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B |$ (Norm Cell Vol)', 
                        R_LIMITS, DBNORM_LIMITS, title)

    plot_NxM(TARGET, base, 'png-dBNorm-r', plots, cols=4, rows=1 )
    
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
                        RHO_LIMITS, DBNORM_LIMITS, 'Day ' + title)
    plots[1] = plotargs(df_day, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                        P_LIMITS, DBNORM_LIMITS, 'Day ' + title)
    plots[2] = plotargs(df_day, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        JMAG_LIMITS, DBNORM_LIMITS, 'Day ' + title)
    plots[3] = plotargs(df_day, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        UMAG_LIMITS, DBNORM_LIMITS, 'Day ' + title)
    plots[4] = plotargs(df_night, 'rho', 'dBmagNorm', True, True, 
                        r'$\rho$', r'$| \delta B |$ (Norm Cell Vol)', 
                        RHO_LIMITS, DBNORM_LIMITS, 'Night ' + title)
    plots[5] = plotargs(df_night, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                        P_LIMITS, DBNORM_LIMITS, 'Night ' + title)
    plots[6] = plotargs(df_night, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        JMAG_LIMITS, DBNORM_LIMITS, 'Night ' + title)
    plots[7] = plotargs(df_night, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                        UMAG_LIMITS, DBNORM_LIMITS, 'Night ' + title)

    plot_NxM(TARGET, base, 'png-dBNorm-various-day-night', plots )
    
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
                        RLOG_LIMITS, DBX_SUM_LIMITS, title)
    plots[1] = plotargs(df_r, 'r', 'dBySum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_y $', 
                        RLOG_LIMITS, DBY_SUM_LIMITS, title)
    plots[2] = plotargs(df_r, 'r', 'dBzSum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_z $', 
                        RLOG_LIMITS, DBZ_SUM_LIMITS, title)
    plots[3] = plotargs(df_r, 'r', 'dBSumMag', True, False, 
                        r'$r/R_E$', r'$| \Sigma_r \delta B |$', 
                        RLOG_LIMITS, DB_SUM_LIMITS, title)
    
    plot_NxM(TARGET, base, 'png-sum-dB-r', plots, cols=4, rows=1 )

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
                    RLOG_LIMITS, DBX_SUM_LIMITS, r'$\parallel$ ' + title)

    plots[4] = plotargs_multiy(df_r, 'r', 
                    ['dBperpendicularxSum', 'dBperpendicularphixSum', 'dBperpendicularphiresxSum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_x (j_{\perp})$',
                    [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                    RLOG_LIMITS, DBX_SUM_LIMITS, r'$\perp$ ' + title)

    plots[1] = plotargs_multiy(df_r, 'r', 
                    ['dBparallelySum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\parallel})$',
                    [r'$\parallel$'], 
                    RLOG_LIMITS, DBY_SUM_LIMITS, r'$\parallel$ ' + title)

    plots[5] = plotargs_multiy(df_r, 'r', 
                    ['dBperpendicularySum', 'dBperpendicularphiySum', 'dBperpendicularphiresySum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\perp})$',
                    [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                    RLOG_LIMITS, DBY_SUM_LIMITS, r'$\perp$ ' + title)

    plots[2] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelzSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\parallel})$',
                     [r'$\parallel$'], 
                     RLOG_LIMITS, DBZ_SUM_LIMITS, r'$\parallel$ ' + title)
    
    plots[6] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularzSum', 'dBperpendicularphizSum', 'dBperpendicularphireszSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\perp})$',
                     [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                     RLOG_LIMITS, DBZ_SUM_LIMITS, r'$\perp$ ' + title)

    plots[3] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\parallel)|$',
                     [r'$\parallel$'], 
                     RLOG_LIMITS, DBZ_SUM_LIMITS, r'$\parallel$ ' + title)
    
    plots[7] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\perp)|$',
                     [r'$\perp$'], 
                     RLOG_LIMITS, DBZ_SUM_LIMITS, r'$\perp$ ' + title)
        
    plot_NxM_multiy(TARGET, base, 'png-sum-dB-para-perp-comp-r', plots, plottype = 'line')

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
                        RLOG_LIMITS, RHO_LIMITS, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'p', True, True, 
                        r'$r$', r'$p$', 
                        RLOG_LIMITS, P_LIMITS, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$', 
                        RLOG_LIMITS, JMAG_LIMITS, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$', 
                        RLOG_LIMITS, UMAG_LIMITS, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'rho', True, True, 
                        r'$r$', r'$\rho$', 
                        RLOG_LIMITS, RHO_LIMITS, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'p', True, True, 
                        r'$r$', r'$p$',  
                        RLOG_LIMITS, P_LIMITS, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$',  
                        RLOG_LIMITS, JMAG_LIMITS, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$',  
                        RLOG_LIMITS, UMAG_LIMITS, 'Night ' + title)

    plot_NxM(TARGET, base, 'png-various-r-day-night', plots )
    
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
                        RLOG_LIMITS, J_LIMITS, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$', 
                        RLOG_LIMITS, J_LIMITS, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$', 
                        RLOG_LIMITS, J_LIMITS, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$', 
                        RLOG_LIMITS, J_LIMITS, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'jx', True, False, 
                        r'$ r/R_E $', r'$j_x$', 
                        RLOG_LIMITS, J_LIMITS, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$',  
                        RLOG_LIMITS, J_LIMITS, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$',  
                        RLOG_LIMITS, J_LIMITS, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$',  
                        RLOG_LIMITS, J_LIMITS, 'Night ' + title)

    plot_NxM(TARGET, base, 'png-jxyz-r-day-night', plots )

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
                        RLOG_LIMITS, U_LIMITS, 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$', 
                        RLOG_LIMITS, U_LIMITS, 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$', 
                        RLOG_LIMITS, U_LIMITS, 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$', 
                        RLOG_LIMITS, U_LIMITS, 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'ux', True, False, 
                        r'$ r/R_E $', r'$u_x$', 
                        RLOG_LIMITS, U_LIMITS, 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$',  
                        RLOG_LIMITS, U_LIMITS, 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$',  
                        RLOG_LIMITS, U_LIMITS, 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$',  
                        RLOG_LIMITS, U_LIMITS, 'Night ' + title)

    plot_NxM(TARGET, base, 'png-uxyz-r-day-night', plots )

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
                    XYZ_LIMITS, J_LIMITS, title)
    plots[1] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$', 
                    XYZ_LIMITS, J_LIMITS, title)
    plots[2] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$', 
                    XYZ_LIMITS, J_LIMITS, title)
    plots[3] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                    XYZ_LIMITS, J_LIMITS, title)
    plots[4] = plotargs(df, coord, 'jr', False, False, 
                    r'$' + coord + '/R_E$', r'$j_r$', 
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    plots[5] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$',  
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    plots[6] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$',  
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    plots[7] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    
    plot_NxM(TARGET, base, 'png-jrtp-'+cut+coord, plots )

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
                    XYZ_LIMITS, J_LIMITS, title)
    plots[1] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$', 
                    XYZ_LIMITS, J_LIMITS, title)
    plots[2] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                    XYZ_LIMITS, J_LIMITS, title)
    plots[4] = plotargs(df, coord, 'jparallelMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\parallel$', 
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    plots[5] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$',  
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    plots[6] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                    XYZ_LIMITS_SMALL, J_LIMITS, title)
    
    plot_NxM(TARGET, base, 'png-jpp-'+cut+coord, plots, cols=3, rows=2 )

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
     
    plots[0] = plotargs(df_jr, 'jr', 'cdf', False, False, 
                    r'$j_r$', r'$CDF$',
                    JCDF_LIMITS, [0,1], title)
    plots[1] = plotargs(df_jtheta, 'jtheta', 'cdf', False, False, 
                    r'$j_\theta$', r'$CDF$',
                    JCDF_LIMITS, [0,1], title)
    plots[2] = plotargs(df_jphi, 'jphi', 'cdf', False, False, 
                    r'$j_\phi$', r'$CDF$',
                    JCDF_LIMITS, [0,1], title)
    
    plot_NxM(TARGET, base, 'png-jrtp-cdf', plots, cols=3, rows=1 )

    return

#############################################################################
#############################################################################
# process_data is the workhorse, it uses the above plot_... functions
# and the BATSRUS_dataframe.py routines to generate various plots
#############################################################################
#############################################################################

def process_data(X, base, dirpath=ORIGIN):
    """Process data in BATSRUS file to create dataframe with calculated quantities.

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
            
        title = title to use in plots, which is derived from base (file basename)
        
        batsrus = BATSRUS data read by swmfio 
    """
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1] + ' (hhmmss)'

    filename = dirpath + base
    df = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df = create_deltaB_rCurrents_dataframe(df, X)

    # logging.info('Creating cumulative sum dB dataframe...')

    # df_r = create_cumulative_sum_dataframe(df)

    # logging.info('Creating dayside/nightside dataframe...')
    # df_day = df[df['x'] >= 0]
    # df_night = df[df['x'] < 0]

    # # Do plots...

    # logging.info('Creating dB (Norm) vs r plots...')
    # plot_db_Norm_r( df, title, base )
    
    # logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    # plot_dBnorm_various_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating cumulative sum B vs r plots...')
    # plot_sum_dB( df_r, title, base )

    # logging.info('Creating cumulative sum B parallel/perpendicular vs r plots...')
    # plot_cumulative_B_para_perp(df_r, title, base)

    # logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    # plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base )

    # logging.info('Creating day /night jx, jy, jz vs r plots...')
    # plot_jx_jy_jz_day_night( df_day, df_night, title, base )

    # logging.info('Creating day/night ux, uy, uz vs r plots...')
    # plot_ux_uy_uz_day_night( df_day, df_night, title, base )

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'x')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'y')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x( df, title, base, coord = 'x')
    # plot_jp_jp_vs_x( df, title, base, coord = 'y')
    # plot_jp_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jrtp CDFs...')
    # df_jr, df_jtheta, df_jphi = create_jrtp_cdf_dataframes(df)
    # plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base)

    return

#############################################################################
#############################################################################
# perform_cuts and perform_not_cuts cut the jr and jphi peaks from the 
# BATSRUS data.  process_data_with_cuts generate plots with the cut data
# removed.  While process_3d_cut_plots generate VTK files containing the
# data cut from BATSRUS.  Use a VTK viewer to see the cut data in 3D.
#############################################################################
#############################################################################

def perform_cuts(df1, title1, cut_selected):
    """perform selected cuts on BATSRUS dataframe, df1.

    Inputs:
        df1 = BATSRUS dataframe on which to make cuts
        
        title1 = base title for plots, will be modified based on cuts
        
        cut_selected = which cut to make
        
    Outputs:
        df2 = dataframe with cuts applied
        
        title2 = title to be used in plots
        
        cutname = string signifying cut
    """

    assert(0 < cut_selected < 4)

    df_tmp = deepcopy(df1)

    # Cut jr peaks, always make this cut
    df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > CUT1_JRMIN].index)
    cutname = r'cut1-jr-'
    title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected > 1):
        # Cut jphi peaks far from earth, which builds on the jr cut above
        # Note, this cuts peaks with r > cut2_rmin
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > CUT2_JPHIMIN, df2['r'] > CUT2_RMIN)].index)
        cutname = 'cut2-jphi-far-' + cutname
        title2 = r'$j_\phi$ Peaks (far) ' + title2

    if(cut_selected > 2):
        # Cut jphi peaks near earth, which builds on the jr and jphi cuts above
        # so it gets the jphi peaks for r <= cut2_rmin
        df2 = df2.drop(df2[df2['jphi'].abs() > CUT3_JPHIMIN].index)
        cutname = 'cut3-jphi-near-' + cutname
        title2 = r'$j_\phi$ Peaks (near) ' + title2

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

    # Isolate jr peaks
    if(cut_selected == 1):
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() <= CUT1_JRMIN].index)
        cutname = 'cut1-jr-'
        title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected == 2):
        # Isolate jphi peaks far from earth, ie., r > cut2_rmin
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > CUT1_JRMIN].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'].abs() <= CUT2_JPHIMIN, df2['r'] <= CUT2_RMIN)].index)
        cutname = 'cut2-jphi-far-'
        title2 = r'$j_\phi$ Peaks (far) ' + title1

    if(cut_selected == 3):
        # Isolate jphi peaks near earth, ie., r <= cut2_rmin
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > CUT1_JRMIN].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > CUT2_JPHIMIN, df2['r'] > CUT2_RMIN)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() <= CUT3_JPHIMIN].index)
        cutname = 'cut3-jphi-near-'
        title2 = r'$j_\phi$ Peaks (near) ' + title1

    if(cut_selected == 4):
        # Isolate residual, that is, anything not isolated above
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > CUT1_JRMIN].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > CUT2_JPHIMIN, df2['r'] > CUT2_RMIN)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() > CUT3_JPHIMIN].index)
        cutname = 'residual-'
        title2 = r'Residual ' + title1

    return df2, title2, cutname

def process_data_with_cuts(X, base, dirpath=ORIGIN, cut_selected=1):
    """Process data in BATSRUS file to create dataframe with calculated quantities.

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
            
        title = title to use in plots, which is derived from base (file basename)
        
        batsrus = BATSRUS data read by swmfio 
    """
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title1 = 'Time: ' + words[1] + ' (hhmmss)'

    # Read BASTRUS file and do delta B calculations
    filename = dirpath + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df1 = create_deltaB_rCurrents_dataframe(df1, X)
    
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

def process_3d_cut_plots(X, base, dirpath=ORIGIN):
    """Process data in BATSRUS file to create 3D plots of points in cuts

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
    Outputs:
        None - other than the saved plot file
    """
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title1 = 'Time: ' + words[1] + ' (hhmmss)'

    filename = dirpath + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df1 = create_deltaB_rCurrents_dataframe(df1, X)

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
    cuts2.write_vtk_to_file( TARGET, base, 'vtk-3d-cut1' )
    
    cuts3 = pointcloud( df3, xyz, colorvars )
    cuts3.convert_to_vtk()
    # cuts3.display_vtk()
    cuts3.write_vtk_to_file( TARGET, base, 'vtk-3d-cut2' )
    
    cuts4 = pointcloud( df4, xyz, colorvars )
    cuts4.convert_to_vtk()
    # cuts4.display_vtk()
    cuts4.write_vtk_to_file( TARGET, base, 'vtk-3d-cut3' )
    
    return

def process_3d_cut_plots2(X, base, dirpath=ORIGIN):
    """Process data in BATSRUS file to create 3D plots of points in cuts

    Inputs:
        X = x,y,z position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        None - other than the saved plot file
    """
    import matplotlib.pyplot as plt
    
    filename = dirpath + base
    df1, title1 = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df1 = create_deltaB_rCurrents_dataframe(df1, X)

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

    # Set some plot configs
    plt.rcParams["figure.figsize"] = [12.8, 7.2]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['axes.grid'] = True
    plt.rcParams['font.size'] = 5

    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=VMIN, vmin=-VMAX, vmax=VMAX)
    cmap = plt.colormaps['coolwarm']

    fig = plt.figure()
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 2, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 3, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 5, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 6, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 7, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_ylim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.set_zlim(PLOT3D_LIMITS[0], PLOT3D_LIMITS[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    plt.tight_layout()

    fig = plt.gcf()
    create_directory(TARGET, 'png-3d-cuts/')
    logging.info(f'Saving {base} 3D cut plot')
    fig.savefig(TARGET + 'png-3d-cuts/' + base + '.out.png-3d-cuts.png')
    plt.close(fig)

    return

#############################################################################
#############################################################################
# The process_sum_db... and the loop_sum_... routines work together to
# generate plots of how the BATSRUS data evolves over time.  The with_cuts
# look at how jr and jphi peaks affect the results
#############################################################################
#############################################################################

def process_sum_db_with_cuts(X, base, dirpath=ORIGIN):
    """Process data in BATSRUS file to create dataframe with calculated quantities,
    but in this case we perform some cuts on the data to isolate high current
    regions.  This cut data is used to determine the fraction of the total B
    field due to each set of currents

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
     Outputs:
        df1 = cumulative sum for original (all) data in north-east-zenith
        
        df1 - df2 = contribution due to points in asym. jr cut in ned
        
        df2 - df3 = contribution due to points in y jphi cut in ned
        
        df3 - df4 = contribution due to points in z jphi cut in ned
    """
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title1 = 'Time: ' + words[1] + ' (hhmmss)'

    filename = dirpath + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df1 = create_deltaB_rCurrents_dataframe(df1, X)

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
    
    timeiso = date_timeISO(base)
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

def process_sum_db(X, base, dirpath=ORIGIN):
    """Process data in BATSRUS file to create dataframe with calculated quantities

    Inputs:
        X = cartesian position where magnetic field will be measured
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
     Outputs:
        df1 = cumulative sum for original (all) data in north-east-zenith
    """

    filename = dirpath + base
    df1 = convert_BATSRUS_to_dataframe(filename, rCurrents=RCURRENTS)
    df1 = create_deltaB_rCurrents_dataframe(df1, X)

    logging.info('Calculate cumulative sums for dataframes for extracted cuts...')

    df1 = create_cumulative_sum_dataframe(df1)
    
    timeiso = date_timeISO(base)
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

def loop_sum_db_with_cuts(X, files):
    """Loop thru data in BATSRUS files to create plots showing the effects of
    various cuts on the data.  See process_data_with_cuts for the specific cuts 
    made

    Inputs:
        X = cartesian position where magnetic field will be measured
        
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
            process_sum_db_with_cuts(X, base=files[i])

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
                        r'Total $B_N$ at (1,0,0)',
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'Uncut')   
        
    plots[1] = plotargs_multiy(df, r'Time (hr)', 
                        ['$j_r$ Total', r'$j_r$ Parallel', r'$j_r$ Perpendicular $\phi$', r'$j_r$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_N$ at (1,0,0)',
                        ['$j_r$ Total', r'$j_r$ Parallel', r'$j_r$ Perpendicular $\phi$', r'$j_r$ Perpendicular Residual'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'Cut 1: $j_r$')   
        
    plots[2] = plotargs_multiy(df, r'Time (hr)', 
                        ['$y j_\phi$ Total', r'$y j_\phi$ Parallel', r'$y j_\phi$ Perpendicular $\phi$', r'$y j_\phi$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_N$ at (1,0,0)',
                        ['$y j_\phi$ Total', r'$y j_\phi$ Parallel', r'$y j_\phi$ Perpendicular $\phi$', r'y $j_\phi$ Perpendicular Residual'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'Cut 2: y $j_\phi$')   
        
    plots[3] = plotargs_multiy(df, r'Time (hr)', 
                        ['$z j_\phi$ Total', r'$z j_\phi$ Parallel', r'$z j_\phi$ Perpendicular $\phi$', r'$z j_\phi$ Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_N$ at (1,0,0)',
                        ['$z j_\phi$ Total', r'$z j_\phi$ Parallel', r'$z j_\phi$ Perpendicular $\phi$', r'z $j_\phi$ Perpendicular Residual'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'Cut 3: z$j_\phi$')   
        
    plots[4] = plotargs_multiy(df, r'Time (hr)', 
                        ['Residual Total', r'Residual Parallel', r'Residual Perpendicular $\phi$', r'Residual Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'Total $B_N$ at (1,0,0)',
                        ['Residual Total', r'Residual Parallel', r'Residual Perpendicular $\phi$', r'Residual Perpendicular Residual'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'Residual')   
        
    plot_NxM_multiy(TARGET, 'para-perp-by-peak', 'parallel-perpendicular-composition', 
                    plots, cols=5, rows=1, plottype = 'line')

    return

def loop_sum_db(X, files):
    """Loop thru data in BATSRUS files to create plots showing the breakdown of
    parallel and perpendicular to B field components

    Inputs:
        X = cartesian position where magnetic field will be assessed
        
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
            process_sum_db(X, base=files[i])

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

    plots = [None] * 4
    
    plots[0] = plotargs_multiy(df, r'Time (hr)', 
                        ['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'$B_N$ at (1,0,0)',
                        ['$B_N(\mathbf{j})$', r'$B_N(\mathbf{j}_\parallel)$', r'$B_N(\mathbf{j}_\perp \cdot \hat \phi)$', r'$B_N(\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi)$'], 
                        TIME_LIMITS, DB_SUM_LIMITS2, r'$B_N$ and components')   
            
    plots[1] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Parallel'], 
                        False, False, 
                        r'Time (hr)',
                        r'$\frac{B_N(\mathbf{j}_\parallel)}{B_N(\mathbf{j})}$ at (1,0,0)',
                        [r'$B_N(\mathbf{j}_\parallel)$'],
                        TIME_LIMITS, FRAC_LIMITS, r'Fraction of $B_N$')
    
    plots[2] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Perpendicular $\phi$'], 
                        False, False, 
                        r'Time (hr)',
                        r'$\frac{B_N(\mathbf{j}_\perp \cdot \hat \phi)}{B_N(\mathbf{j})}$ at (1,0,0)',
                        [r'$B_N(\mathbf{j}_\perp \cdot \hat \phi)$'], 
                        TIME_LIMITS, FRAC_LIMITS, r'Fraction of $B_N$')
    
    plots[3] = plotargs_multiy(df, r'Time (hr)', 
                        [r'Fraction Perpendicular Residual'], 
                        False, False, 
                        r'Time (hr)',
                        r'$\frac{B_N(\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi)}{B_N(\mathbf{j})}$ at (1,0,0)',
                        [r'$B_N(\mathbf{j}_\perp - \mathbf{j}_\perp \cdot \hat \phi)$'], 
                        TIME_LIMITS, FRAC_LIMITS, r'Fraction of $B_N$')
    
    plot_NxM_multiy(TARGET, 'para-perp', 'parallel-perpendicular-composition', 
                    plots, cols=4, rows=1, plottype = 'line')
    
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

    X = [1, 0, 0]

    process_data(X, inputfile)
    # process_data_with_cuts(X, inputfile, cut_selected = 3)
    # process_3d_cut_plots(X, inputfile)
    # process_3d_cut_plots2(X, inputfile)


def main2(argv):
    if COLABA:
        files = get_files(ORIGIN, reduce=True, base='3d__var_2_e*')
    else:
        files = get_files(ORIGIN, base='3d__*')
        
    logging.info('Num. of files: ' + str(len(files)))

    X = [1, 0, 0]

    # loop_sum_db_with_cuts(X, files)
    loop_sum_db(X, files)
    
if __name__ == "__main__":
   main2(sys.argv[1:])


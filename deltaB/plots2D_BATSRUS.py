#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022
@author: Dean Thomas
"""
import logging
import os.path
from copy import deepcopy
import numpy as np

from deltaB.plotting import plotargs, plotargs_multiy, \
    plot_NxM, plot_NxM_multiy, pointcloud
from deltaB.BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_deltaB_spherical_dataframe, \
    create_deltaB_rCurrents_spherical_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_cumulative_sum_dataframe, \
    create_cumulative_sum_spherical_dataframe, \
    create_jrtp_cdf_dataframes
from deltaB.util import create_directory

# info = {...} example is below

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

# limits = {...} example is below

# Range of values seen in each variable, used to plot graphs
# limits = { 
#     "RLOG_LIMITS": [1, 1000],
#     "R_LIMITS": [0, 300],
#     'RHO_LIMITS': [10**-2, 10**4],
#    'P_LIMITS': [10**-5, 10**3],
#     'JMAG_LIMITS': [10**-11, 10**1],
#     'J_LIMITS': [-1, 1],
#     'limits['JCDF_LIMITS']': [-0.1, 0.1],
#     'UMAG_LIMITS': [10**-3, 10**4],
#     'U_LIMITS': [-1100, 1100],
#     'DBNORM_LIMITS': [10**-15, 10**-1],
#     'DBX_SUM_LIMITS': [-1500, 1500],
#     'DBY_SUM_LIMITS': [-1500, 1500],
#     'DBZ_SUM_LIMITS': [-1500, 1500],
#     'DBP_SUM_LIMITS': [-1500, 1500],
#     'DB_SUM_LIMITS': [0, 1500],
#     'DB_SUM_LIMITS2': [-1200,400],
#     'limits['PLOT3D_LIMITS']': [-10, 10],
#     'XYZ_LIMITS': [-300, 300],
#     'XYZ_LIMITS_SMALL': [-20, 20],
#     'TIME_LIMITS': [4,16],
#     'FRAC_LIMITS': [-0.5,1.5],
#     'VMIN': 0.02,
#     'VMAX': 0.5
# }

# cuts = {...} is below

# # Range of values for cuts
# cuts = {
#     'cuts['CUT1_JRMIN']': 0.02,
#     'cuts['CUT2_JPHIMIN']': 0.02,
#     'cuts['CUT2_RMIN']': 5,
#     'cuts['CUT3_JPHIMIN']': 0.02
# }

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

def plot_db_Norm_r(df, title, base, info, limits):
    """Plot components and magnitude of dB in each cell versus radius r.
    In this procedure, the dB values are normalized by cell volume
    Inputs:
        df = dataframe with BATSRUS data and calculated variables
        
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """
     
    plots = [None] * 4
    
    plots[0] = plotargs(df, 'r', 'dBxNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_x |$ (Norm Cell Vol)', 
                       limits['R_LIMITS'],limits['DBNORM_LIMITS'], title)
    plots[1] = plotargs(df, 'r', 'dByNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_y |$ (Norm Cell Vol)', 
                       limits['R_LIMITS'],limits['DBNORM_LIMITS'], title)
    plots[2] = plotargs(df, 'r', 'dBzNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B_z |$ (Norm Cell Vol)', 
                       limits['R_LIMITS'],limits['DBNORM_LIMITS'], title)
    plots[3] = plotargs(df, 'r', 'dBmagNorm', False, True, 
                        r'$r/R_E$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['R_LIMITS'],limits['DBNORM_LIMITS'], title)

    plot_NxM(info['dir_plots'], base, 'dBNorm-r', plots, cols=4, rows=1 )
    
    return

def plot_dBnorm_various_day_night(df_day, df_night, title, base, info, limits):
    """Plot dBmagNorm vs rho, p, magnitude of j, and magnitude of u in each cell.  
    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
            
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
            
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'rho', 'dBmagNorm', True, True, 
                        r'$\rho$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['RHO_LIMITS'],limits['DBNORM_LIMITS'], 'Day ' + title)
    plots[1] = plotargs(df_day, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['P_LIMITS'],limits['DBNORM_LIMITS'], 'Day ' + title)
    plots[2] = plotargs(df_day, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['JMAG_LIMITS'],limits['DBNORM_LIMITS'], 'Day ' + title)
    plots[3] = plotargs(df_day, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['UMAG_LIMITS'],limits['DBNORM_LIMITS'], 'Day ' + title)
    plots[4] = plotargs(df_night, 'rho', 'dBmagNorm', True, True, 
                        r'$\rho$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['RHO_LIMITS'],limits['DBNORM_LIMITS'], 'Night ' + title)
    plots[5] = plotargs(df_night, 'p', 'dBmagNorm', True, True, 
                        r'$p$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['P_LIMITS'],limits['DBNORM_LIMITS'], 'Night ' + title)
    plots[6] = plotargs(df_night, 'jMag', 'dBmagNorm', True, True, 
                        r'$| j |$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['JMAG_LIMITS'],limits['DBNORM_LIMITS'], 'Night ' + title)
    plots[7] = plotargs(df_night, 'uMag', 'dBmagNorm', True, True, 
                        r'$| u |$', r'$| \delta B |$ (Norm Cell Vol)', 
                       limits['UMAG_LIMITS'],limits['DBNORM_LIMITS'], 'Night ' + title)

    plot_NxM(info['dir_plots'], base, 'dBNorm-various-day-night', plots )
    
    return

def plot_sum_dB(df_r, title, base, info, limits):
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
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """
    plots = [None] * 4
    
    plots[0] = plotargs(df_r, 'r', 'dBxSum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_x $', 
                       limits['RLOG_LIMITS'],limits['DBX_SUM_LIMITS'], title)
    plots[1] = plotargs(df_r, 'r', 'dBySum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_y $', 
                       limits['RLOG_LIMITS'],limits['DBY_SUM_LIMITS'], title)
    plots[2] = plotargs(df_r, 'r', 'dBzSum', True, False, 
                        r'$r/R_E$', r'$\Sigma_r \delta B_z $', 
                       limits['RLOG_LIMITS'],limits['DBZ_SUM_LIMITS'], title)
    plots[3] = plotargs(df_r, 'r', 'dBSumMag', True, False, 
                        r'$r/R_E$', r'$| \Sigma_r \delta B |$', 
                       limits['RLOG_LIMITS'],limits['DB_SUM_LIMITS'], title)
    
    plot_NxM(info['dir_plots'], base, 'sum-dB-r', plots, cols=4, rows=1 )

    return

def plot_cumulative_B_para_perp(df_r, title, base, info, limits):
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
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs_multiy(df_r, 'r', 
                    ['dBparallelxSum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_x (j_{\parallel})$',
                    [r'$\parallel$'], 
                    limits['RLOG_LIMITS'],limits['DBX_SUM_LIMITS'], r'$\parallel$ ' + title)

    plots[4] = plotargs_multiy(df_r, 'r', 
                    ['dBperpendicularxSum', 'dBperpendicularphixSum', 'dBperpendicularphiresxSum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_x (j_{\perp})$',
                    [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                    limits['RLOG_LIMITS'],limits['DBX_SUM_LIMITS'], r'$\perp$ ' + title)

    plots[1] = plotargs_multiy(df_r, 'r', 
                    ['dBparallelySum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\parallel})$',
                    [r'$\parallel$'], 
                    limits['RLOG_LIMITS'],limits['DBY_SUM_LIMITS'], r'$\parallel$ ' + title)

    plots[5] = plotargs_multiy(df_r, 'r', 
                    ['dBperpendicularySum', 'dBperpendicularphiySum', 'dBperpendicularphiresySum'], 
                    True, False, 
                    r'$r/R_E$', r'$\Sigma_r \delta B_y (j_{\perp})$',
                    [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                    limits['RLOG_LIMITS'],limits['DBY_SUM_LIMITS'], r'$\perp$ ' + title)

    plots[2] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelzSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\parallel})$',
                     [r'$\parallel$'], 
                     limits['RLOG_LIMITS'],limits['DBZ_SUM_LIMITS'], r'$\parallel$ ' + title)
    
    plots[6] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularzSum', 'dBperpendicularphizSum', 'dBperpendicularphireszSum'], 
                     True, False, 
                     r'$r/R_E$', r'$\Sigma_r \delta B_z (j_{\perp})$',
                     [r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'], 
                     limits['RLOG_LIMITS'],limits['DBZ_SUM_LIMITS'], r'$\perp$ ' + title)

    plots[3] = plotargs_multiy(df_r, 'r', 
                     ['dBparallelSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\parallel)|$',
                     [r'$\parallel$'], 
                     limits['RLOG_LIMITS'],limits['DBZ_SUM_LIMITS'], r'$\parallel$ ' + title)
    
    plots[7] = plotargs_multiy(df_r, 'r', 
                     ['dBperpendicularSumMag'], 
                     True, False, 
                     r'$r/R_E$', r'$| \Sigma_r \delta B (j_\perp)|$',
                     [r'$\perp$'], 
                     limits['RLOG_LIMITS'],limits['DBZ_SUM_LIMITS'], r'$\perp$ ' + title)
        
    plot_NxM_multiy(info['dir_plots'], base, 'sum-dB-para-perp-comp-r', plots, plottype = 'scatter')

    return

def plot_rho_p_jMag_uMag_day_night(df_day, df_night, title, base, info, limits):
    """Plot rho, p, magnitude of j, and magnitude of u in each cell versus 
        radius r.  
    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
            
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
            
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """
    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'rho', True, True, 
                        r'$r$', r'$\rho$', 
                       limits['RLOG_LIMITS'],limits['RHO_LIMITS'], 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'p', True, True, 
                        r'$r$', r'$p$', 
                       limits['RLOG_LIMITS'],limits['P_LIMITS'], 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$', 
                       limits['RLOG_LIMITS'],limits['JMAG_LIMITS'], 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$', 
                       limits['RLOG_LIMITS'],limits['UMAG_LIMITS'], 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'rho', True, True, 
                        r'$r$', r'$\rho$', 
                       limits['RLOG_LIMITS'],limits['RHO_LIMITS'], 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'p', True, True, 
                        r'$r$', r'$p$',  
                       limits['RLOG_LIMITS'],limits['P_LIMITS'], 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jMag', True, True, 
                        r'$r$', r'$| j |$',  
                       limits['RLOG_LIMITS'],limits['JMAG_LIMITS'], 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, True, 
                        r'$r$', r'$| u |$',  
                       limits['RLOG_LIMITS'],limits['UMAG_LIMITS'], 'Night ' + title)

    plot_NxM(info['dir_plots'], base, 'various-r-day-night', plots )
    
    return

def plot_jx_jy_jz_day_night(df_day, df_night, title, base, info, limits):
    """Plot jx, jy, jz  in each cell versus radius r.  
    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
            
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
            
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'jx', True, False, 
                        r'$ r/R_E $', r'$j_x$',
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$', 
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$', 
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$', 
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'jx', True, False, 
                        r'$ r/R_E $', r'$j_x$', 
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'jy', True, False, 
                        r'$ r/R_E $', r'$j_y$',  
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'jz', True, False, 
                        r'$ r/R_E $', r'$j_z$',  
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'jMag', True, False, 
                        r'$ r/R_E $', r'$| j |$',  
                       limits['RLOG_LIMITS'],limits['J_LIMITS'], 'Night ' + title)

    plot_NxM(info['dir_plots'], base, 'jxyz-r-day-night', plots )

    return

def plot_ux_uy_uz_day_night(df_day, df_night, title, base, info, limits):
    """Plot ux, uy, uz  in each cell versus radius r.  
    Inputs:
        df_day = dataframe containing r, rho, jMag, uMag for day side of earth,
            x >= 0
            
        df_night = dataframe containing r, rho, jMag, uMag for night side of 
            earth, x < 0
            
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df_day, 'r', 'ux', True, False, 
                        r'$ r/R_E $', r'$u_x$',
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Day ' + title)
    plots[1] = plotargs(df_day, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$', 
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Day ' + title)
    plots[2] = plotargs(df_day, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$', 
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Day ' + title)
    plots[3] = plotargs(df_day, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$', 
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Day ' + title)
    plots[4] = plotargs(df_night, 'r', 'ux', True, False, 
                        r'$ r/R_E $', r'$u_x$', 
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Night ' + title)
    plots[5] = plotargs(df_night, 'r', 'uy', True, False, 
                        r'$ r/R_E $', r'$u_y$',  
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Night ' + title)
    plots[6] = plotargs(df_night, 'r', 'uz', True, False, 
                        r'$ r/R_E $', r'$u_z$',  
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Night ' + title)
    plots[7] = plotargs(df_night, 'r', 'uMag', True, False, 
                        r'$ r/R_E $', r'$| u |$',  
                       limits['RLOG_LIMITS'],limits['U_LIMITS'], 'Night ' + title)

    plot_NxM(info['dir_plots'], base, 'uxyz-r-day-night', plots )

    return

def plot_jr_jt_jp_vs_x(df, title, base, info, limits, coord='x', cut='', ):
    """Plot jr, jtheta, jphi  in each cell versus x.  
    Inputs:
        df = dataframe containing jr, jtheta, jphi and x
        
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
        coord = which coordinate is on the x axis - x, y, or z
        
        cut = which cut was made, used in plot filenames
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df, coord, 'jr', False, False, 
                    r'$' + coord + '/R_E$', r'$j_r$',
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[1] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$', 
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[2] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$', 
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[3] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[4] = plotargs(df, coord, 'jr', False, False, 
                    r'$' + coord + '/R_E$', r'$j_r$', 
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    plots[5] = plotargs(df, coord, 'jtheta', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\theta$',  
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    plots[6] = plotargs(df, coord, 'jphi', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\phi$',  
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    plots[7] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    
    plot_NxM(info['dir_plots'], base, 'jrtp-'+cut+coord, plots )

    return

def plot_jp_jp_vs_x(df, title, base, info, limits, coord='x', cut=''):
    """Plot jparallel and jperpendicular to the B field in each cell versus x.  
    Inputs:
        df = dataframe containing jparallel, jperpendicular, and x
        
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
        coord = which coordinate is on the x axis - x, y, or z
        
        cut = which cut was made, used in plot filenames
        
    Outputs:
        None 
     """

    plots = [None] * 8
    
    plots[0] = plotargs(df, coord, 'jparallelMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\parallel$',
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[1] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$', 
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[2] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$', 
                   limits['XYZ_LIMITS'],limits['J_LIMITS'], title)
    plots[4] = plotargs(df, coord, 'jparallelMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\parallel$', 
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    plots[5] = plotargs(df, coord, 'jperpendicularMag', False, False, 
                    r'$' + coord + '/R_E$', r'$j_\perp$',  
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    plots[6] = plotargs(df, coord, 'jMag', False, False, 
                    r'$' + coord + '/R_E$', r'$| j |$',  
                   limits['XYZ_LIMITS_SMALL'],limits['J_LIMITS'], title)
    
    plot_NxM(info['dir_plots'], base, 'jpp-'+cut+coord, plots, cols=4, rows=2 )

    return

def plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base, info, limits):
    """Plot jr, jtheta, jphi CDFs  
    Inputs:
        df_jr, df_jtheta, df_jphi = dataframe CDF data
        
        title = title for plots
        
        base = basename of file where plot will be stored
        
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None 
     """

    plots = [None] * 3
     
    plots[0] = plotargs(df_jr, 'jr', 'cdf', False, False, 
                    r'$j_r$', r'$CDF$',
                    limits['JCDF_LIMITS'], [0,1], title)
    plots[1] = plotargs(df_jtheta, 'jtheta', 'cdf', False, False, 
                    r'$j_\theta$', r'$CDF$',
                    limits['JCDF_LIMITS'], [0,1], title)
    plots[2] = plotargs(df_jphi, 'jphi', 'cdf', False, False, 
                    r'$j_\phi$', r'$CDF$',
                    limits['JCDF_LIMITS'], [0,1], title)
    
    plot_NxM(info['dir_plots'], base, 'jrtp-cdf', plots, cols=3, rows=1 )

    return

#############################################################################
#############################################################################
# process_BATSRUS is the workhorse, it uses the above plot_... functions
# and the BATSRUS_dataframe.py routines to generate various plots
#############################################################################
#############################################################################

def process_BATSRUS(XGSM, filepath, time, info, limits):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
    Inputs:
        X = cartesian position where magnetic field will be measured (GSM coordinates)
        
        time = time associated with each file within info.  Used as a key to 
            get filepath
            
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
    Outputs:
        None = other than plots saved to files
        
    """
    # We need the filepath base for BATSRUS file
    base = os.path.basename(filepath)

    # Create the title that we'll use in the graphics
    # title = 'Time: ' + str(time[3]).zfill(2) + str(time[4]).zfill(2) + \
    #     str(time[5]).zfill(2) + ' (hhmmss)'
    title = str(time[0]) + '-' + str(time[1]).zfill(2) + '-' + str(time[2]).zfill(2) + 'T' + \
            str(time[3]).zfill(2) +':' + str(time[4]).zfill(2) + ':' + str(time[5]).zfill(2)

    df = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    df = create_deltaB_spherical_dataframe(df)
    df = create_deltaB_rCurrents_dataframe(df, XGSM)
    df = create_deltaB_rCurrents_spherical_dataframe(df, XGSM)
    

    logging.info('Creating cumulative sum dB dataframe...')

    df_r = create_cumulative_sum_dataframe(df)
    df_sph = create_cumulative_sum_spherical_dataframe(df)

    logging.info('Creating dayside/nightside dataframe...')
    df_day = df[df['x'] >= 0]
    df_night = df[df['x'] < 0]

    # Do plots...

    logging.info('Creating dB (Norm) vs r plots...')
    plot_db_Norm_r( df, title, base, info, limits )
    
    logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    plot_dBnorm_various_day_night( df_day, df_night, title, base, info, limits )
    
    logging.info('Creating cumulative sum B vs r plots...')
    plot_sum_dB( df_r, title, base, info, limits )

    # logging.info('Creating cumulative sum B parallel/perpendicular vs r plots...')
    # plot_cumulative_B_para_perp(df_sph, title, base, info, limits)

    logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base, info, limits )

    logging.info('Creating day /night jx, jy, jz vs r plots...')
    plot_jx_jy_jz_day_night( df_day, df_night, title, base, info, limits )

    logging.info('Creating day/night ux, uy, uz vs r plots...')
    plot_ux_uy_uz_day_night( df_day, df_night, title, base, info, limits )

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x( df_sph, title, base, info, limits, coord = 'x')
    plot_jr_jt_jp_vs_x( df_sph, title, base, info, limits, coord = 'y')
    plot_jr_jt_jp_vs_x( df_sph, title, base, info, limits, coord = 'z')

    logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    plot_jp_jp_vs_x( df, title, base, info, limits, coord = 'x')
    plot_jp_jp_vs_x( df, title, base, info, limits, coord = 'y')
    plot_jp_jp_vs_x( df, title, base, info, limits, coord = 'z')

    logging.info('Creating jrtp CDFs...')
    df_jr, df_jtheta, df_jphi = create_jrtp_cdf_dataframes(df)
    plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base, info, limits)

    return

#############################################################################
#############################################################################
# perform_cuts and perform_not_cuts cut the jr and jphi peaks from the 
# BATSRUS data.  process_data_with_cuts generate plots with the cut data
# removed.  While process_3d_cut_plots generate VTK files containing the
# data cut from BATSRUS.  Use a VTK viewer to see the cut data in 3D.
#############################################################################
#############################################################################

def perform_cuts(df1, title1, cuts, cut_selected):
    """perform selected cuts on BATSRUS dataframe, df1.
    Inputs:
        df1 = BATSRUS dataframe on which to make cuts
        
        title1 = base title for plots, will be modified based on cuts
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        cut_selected = which cut, of those specified in cuts, is to be performed
        
    Outputs:
        df2 = dataframe with cuts applied
        
        title2 = title to be used in plots
        
        cutname = string signifying cut
    """

    assert(0 < cut_selected < 4)

    df_tmp = deepcopy(df1)

    # Cut jr peaks, always make this cut
    df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cuts['CUT1_JRMIN']].index)
    cutname = r'cut1-jr-'
    title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected > 1):
        # Cut jphi peaks far from earth, which builds on the jr cut above
        # Note, this cuts peaks with r > cuts['CUT2_RMIN']
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cuts['CUT2_JPHIMIN'], df2['r'] > cuts['CUT2_RMIN'])].index)
        cutname = 'cut2-jphi-far-' + cutname
        title2 = r'$j_\phi$ Peaks (far) ' + title2

    if(cut_selected > 2):
        # Cut jphi peaks near earth, which builds on the jr and jphi cuts above
        # so it gets the jphi peaks for r <= cuts['CUT2_RMIN']
        df2 = df2.drop(df2[df2['jphi'].abs() > cuts['CUT3_JPHIMIN']].index)
        cutname = 'cut3-jphi-near-' + cutname
        title2 = r'$j_\phi$ Peaks (near) ' + title2

    return df2, title2, cutname

def perform_not_cuts(df1, title1, cuts, cut_selected):
    """perform selected cuts on BATSRUS dataframe, df1. Creates the opposite result
    of perform_cuts
    Inputs:
        df1 = BATSRUS dataframe on which to make cuts

        title1 = base title for plots, will be modified based on cuts
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        cut_selected = which cut, of those specified in cuts, is to be performed
        
    Outputs:
        df2 = dataframe with cuts applied
        
        title2 = title to be used in plots
        
        cutname = string signifying cut
    """

    assert(0 < cut_selected < 5)

    df_tmp = deepcopy(df1)

    # Isolate jr peaks
    if(cut_selected == 1):
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() <= cuts['CUT1_JRMIN']].index)
        cutname = 'cut1-jr-'
        title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected == 2):
        # Isolate jphi peaks far from earth, ie., r > cuts['CUT2_RMIN']
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cuts['CUT1_JRMIN']].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'].abs() <= cuts['CUT2_JPHIMIN'], df2['r'] <= cuts['CUT2_RMIN'])].index)
        cutname = 'cut2-jphi-far-'
        title2 = r'$j_\phi$ Peaks (far) ' + title1

    if(cut_selected == 3):
        # Isolate jphi peaks near earth, ie., r <= cuts['CUT2_RMIN']
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cuts['CUT1_JRMIN']].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cuts['CUT2_JPHIMIN'], df2['r'] > cuts['CUT2_RMIN'])].index)
        df2 = df2.drop(df2[df2['jphi'].abs() <= cuts['CUT3_JPHIMIN']].index)
        cutname = 'cut3-jphi-near-'
        title2 = r'$j_\phi$ Peaks (near) ' + title1

    if(cut_selected == 4):
        # Isolate residual, that is, anything not isolated above
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cuts['CUT1_JRMIN']].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cuts['CUT2_JPHIMIN'], df2['r'] > cuts['CUT2_RMIN'])].index)
        df2 = df2.drop(df2[df2['jphi'].abs() > cuts['CUT3_JPHIMIN']].index)
        cutname = 'residual-'
        title2 = r'Residual ' + title1

    return df2, title2, cutname

#############################################################################
#############################################################################
# process_BATSRUS_with_cuts is a workhorse, it uses the above perform_not_cuts
# and the BATSRUS_dataframe.py routines to generate various plots isolating
# specific cuts on the BATSRUS data
#############################################################################
#############################################################################

def process_BATSRUS_with_cuts(XGSM, filepath, time, info, limits, cuts, cut_selected=1):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
    Making cuts to the data as specified in cuts = {...}.
    
    Inputs:
        X = cartesian position where magnetic field will be measured (GSM coordinates)
        
        time = time associated with each file within info.  Used as a key to 
            get filepath
            
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        cut_selected = which cut, of those specified in cuts, is to be performed
        
    Outputs:
        None = other than the pltos saved to files
    """
    # We need the filepath base for BATSRUS file
    base = os.path.basename(filepath)

    # Create the title that we'll use in the graphics
    # title = 'Time: ' + str(time[3]).zfill(2) + str(time[4]).zfill(2) + \
    #     str(time[5]).zfill(2) + ' (hhmmss)'
    title1 = str(time[0]) + '-' + str(time[1]).zfill(2) + '-' + str(time[2]).zfill(2) + 'T' + \
            str(time[3]).zfill(2) +':' + str(time[4]).zfill(2) + ':' + str(time[5]).zfill(2)

    df1 = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    df1 = create_deltaB_spherical_dataframe(df1)
    df1 = create_deltaB_rCurrents_dataframe(df1, XGSM)
    df1 = create_deltaB_rCurrents_spherical_dataframe(df1, XGSM)
    
    # Perform cuts on BATSRUS data
    df2, title2, cutname = perform_cuts(df1, title1, cuts, cut_selected=cut_selected)

    # Do plots...

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x( df2, title2, base, info, limits, coord = 'x', cut=cutname)
    plot_jr_jt_jp_vs_x( df2, title2, base, info, limits, coord = 'y', cut=cutname)
    plot_jr_jt_jp_vs_x( df2, title2, base, info, limits, coord = 'z', cut=cutname)

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x( df2, title2, base, info, limits, coord = 'x', cut=cutname)
    # plot_jp_jp_vs_x( df2, title2, base, info, limits, coord = 'y', cut=cutname)
    # plot_jp_jp_vs_x( df2, title2, base, info, limits, coord = 'z', cut=cutname)

    return

def process_BATSRUS_3d_cut_vtk(XGSM, filepath, time, info, limits, cuts):
    """Process data in BATSRUS file to create 3D plots of points in cuts.  Make
    cuts to the data as specified in cuts = {...}.
    
    Inputs:
        X = cartesian position where magnetic field will be measured (GSM coordinates)
        
        time = time associated with each file within info.  Used as a key to 
            get filepath
            
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
    Outputs:
        None = other than the plots saved to files
    """

    # We need the filepath base for BATSRUS file
    base = os.path.basename(filepath)

    # Create the title that we'll use in the graphics
    # title = 'Time: ' + str(time[3]).zfill(2) + str(time[4]).zfill(2) + \
    #     str(time[5]).zfill(2) + ' (hhmmss)'
    title1 = str(time[0]) + '-' + str(time[1]).zfill(2) + '-' + str(time[2]).zfill(2) + 'T' + \
            str(time[3]).zfill(2) +':' + str(time[4]).zfill(2) + ':' + str(time[5]).zfill(2)

    df1 = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    df1 = create_deltaB_spherical_dataframe(df1)
    df1 = create_deltaB_rCurrents_dataframe(df1, XGSM)
    df1 = create_deltaB_rCurrents_spherical_dataframe(df1, XGSM)

    logging.info('Creating dataframes with extracted cuts...')

    df2, title2, cutname2 = perform_not_cuts(df1, title1, cuts, cut_selected=1)
    df3, title3, cutname3 = perform_not_cuts(df1, title1, cuts, cut_selected=2)
    df4, title4, cutname4 = perform_not_cuts(df1, title1, cuts, cut_selected=3)
 
    logging.info('Plotting 3D extracted cuts...')

    xyz = ['x', 'y', 'z']
    colorvars = ['jr', 'jtheta', 'jphi', 'jparallelMag', 'jperpendicularMag']
    
    logging.info(f'Saving {base} 3D cut plots')

    cuts2 = pointcloud( df2, xyz, colorvars )
    cuts2.convert_to_vtk()
    # cuts2.display_vtk()
    cuts2.write_vtk_to_file( info['dir_plots'], base, 'vtk-3d-cut1' )
    
    cuts3 = pointcloud( df3, xyz, colorvars )
    cuts3.convert_to_vtk()
    # cuts3.display_vtk()
    cuts3.write_vtk_to_file( info['dir_plots'], base, 'vtk-3d-cut2' )
    
    cuts4 = pointcloud( df4, xyz, colorvars )
    cuts4.convert_to_vtk()
    # cuts4.display_vtk()
    cuts4.write_vtk_to_file( info['dir_plots'], base, 'vtk-3d-cut3' )
    
    return

def process_BATSRUS_3d_cut_plots(XGSM, filepath, time, info, limits, cuts):
    """Process data in BATSRUS file to create 3D plots of points in cuts.  Make
    cuts to the data as specified in cuts = {...}.
    
    Inputs:
        X = cartesian position where magnetic field will be measured (GSM coordinates)
        
        time = time associated with each file within info.  Used as a key to 
            get filepath
            
        info = info on files to be processed, see info = {...} example above
        
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
    Outputs:
        None = other than the plots saved to files
    """
    import matplotlib.pyplot as plt
    
    # We need the filepath base for BATSRUS file
    base = os.path.basename(filepath)

    # Create the title that we'll use in the graphics
    # title = 'Time: ' + str(time[3]).zfill(2) + str(time[4]).zfill(2) + \
    #     str(time[5]).zfill(2) + ' (hhmmss)'
    title1 = str(time[0]) + '-' + str(time[1]).zfill(2) + '-' + str(time[2]).zfill(2) + 'T' + \
            str(time[3]).zfill(2) +':' + str(time[4]).zfill(2) + ':' + str(time[5]).zfill(2)

    df1 = convert_BATSRUS_to_dataframe(filepath, info['rCurrents'])
    df1 = create_deltaB_spherical_dataframe(df1)
    df1 = create_deltaB_rCurrents_dataframe(df1, XGSM)
    df1 = create_deltaB_rCurrents_spherical_dataframe(df1, XGSM)

    #################################
    #################################
    # Cut asymmetric jr vs y lobes
    df2, title2, cutname2 = perform_not_cuts(df1, title1, cuts, cut_selected=1)
    #################################
    #################################
    # Cut jphi vs y blob
    df3, title3, cutname3 = perform_not_cuts(df1, title1, cuts, cut_selected=2)
    #################################
    #################################
    # Cut jphi vs z blob
    df4, title4, cutname4 = perform_not_cuts(df1, title1, cuts,cut_selected=3)
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
    norm = SymLogNorm(linthresh=limits['VMIN'], vmin=-limits['VMAX'], vmax=limits['VMAX'])
    cmap = plt.colormaps['coolwarm']

    fig = plt.figure()
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 2, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 3, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 5, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 6, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 7, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_ylim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.set_zlim(limits['PLOT3D_LIMITS'][0], limits['PLOT3D_LIMITS'][1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    plt.tight_layout()

    fig = plt.gcf()
    create_directory(info['dir_plots'], '3d-cuts/')

    name = base + '.3d-cuts.png'
    filename = os.path.join( info['dir_plots'], '3d-cuts', name )

    logging.info(f'Saving {base} 3D cut plot... {name}')
    fig.savefig(filename)
    plt.close(fig)

    return

def loop_2D_BATSRUS(XGSM, info, limits, reduce):
    """Loop thru data in BATSRUS files to create a wide variety of 2D plots 
    showing various parameters.  process_BATSRUS has details of the plots.
    
    Inputs:
        XGSM = cartesian position where magnetic field will be assessed (GSM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        limits = axis limits on plots, see limits = {...} example above
        
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        None - other than the pickle file saved
    """

    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
 
        logging.info(f'Plots for BASTRUS file... {base}')
    
        # Process file
        process_BATSRUS(XGSM, filepath, time, info, limits)

    return

def loop_2D_BATSRUS_with_cuts(XGSM, info, limits, cuts, cut_selected, reduce):
    """Loop thru data in BATSRUS files to create a wide variety of 2D plots 
    showing various parameters.  process_BATSRUS_with_cuts has details of the plots.
    These plots are the same as those produced by process_BATSRUS except that
    cuts have been made to the data to highlight regions in the magnetosphere.
    
    Inputs:
        XGSM = cartesian position where magnetic field will be assessed (GSM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        cut_selected = cut selected from those in cuts = {...}
        
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        None - other than the plot files saved
    """

    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
 
        logging.info(f'Plots for BASTRUS file... {base}')
    
        # Process file
        process_BATSRUS_with_cuts(XGSM, filepath, time, info, limits, cuts, cut_selected = cut_selected)

    return

def loop_2D_BATSRUS_3d_cut_vtk(XGSM, info, limits, cuts, reduce):
    """Loop thru data in BATSRUS files to create a wide variety of 2D plots 
    showing various parameters.  process_BATSRUS_3d_cut_vtk has details of the 
    These are 3d VTK plots highlighting regions of the magnetosphere identified
    through the cuts.
    
    Inputs:
        XGSM = cartesian position where magnetic field will be assessed (GSM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        None - other than the vtk plot files 
    """

    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
 
        logging.info(f'Plots for BASTRUS file... {base}')
    
        # Process file
        process_BATSRUS_3d_cut_vtk(XGSM, filepath, time, info, limits, cuts)
        # process_BATSRUS_3d_cut_plots(XGSM, filepath, time, info, limits, cuts)

    return

def loop_2D_BATSRUS_3d_cut_plots(XGSM, info, limits, cuts, reduce):
    """Loop thru data in BATSRUS files to create a wide variety of 2D plots 
    showing various parameters.  process_BATSRUS_3d_cut_vtk has details of the 
    These are 3d matplotlib plots highlighting regions of the magnetosphere 
    identified through the cuts.
    
    Inputs:
        XGSM = cartesian position where magnetic field will be assessed (GSM coordinates)
        
        info = info on files to be processed, see info = {...} example above
             
        limits = axis limits on plots, see limits = {...} example above
        
        cuts = cuts made to the data, see cuts = {...} example above
        
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
    Outputs:
        None - other than the vtk plot files 
    """

    # Time associated with each file
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Loop through the files and process them
    for i in range(n):   
        time = times[i]
        
        # We need the filepath for RIM file
        filepath = info['files']['magnetosphere'][time]
        base = os.path.basename(filepath)
 
        logging.info(f'Plots for BASTRUS file... {base}')
    
        # Process file
        process_BATSRUS_3d_cut_plots(XGSM, filepath, time, info, limits, cuts)

    return

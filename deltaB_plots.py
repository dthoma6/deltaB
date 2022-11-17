#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: dean
"""
# Info on divB_simple1 runs available at:
#
# https://ccmc.gsfc.nasa.gov/results/viewrun.php?domain=GM&runnumber=Brian_Curtis_042213_7
#
# https://ccmc.gsfc.nasa.gov/RoR_WWW/GM/SWMF/2013/Brian_Curtis_042213_7/Brian_Curtis_042213_7_sw.gif
#
 


# origin and target define where input data and output plots are stored
origin = '/Volumes/Physics HD v2/divB_simple1/GM/'
target = '/Volumes/Physics HD v2/divB_simple1/plots/'

# rCurrents define range from earth center below which results are not valid
# measured in Re units
rCurrents = 1.8

# Initialize useful variables
(X,Y,Z) = (1.0, 0.0, 0.0) 
  
# range of values seen in each variable, used to plot graphs
rho_limits    = [10**-2,10**2]
p_limits      = [10**-5,10**2]
jMag_limits   = [10**-11,10**0]
j_limits      = [-0.3,0.3]    
uMag_limits   = [10**-3,10**4]
u_limits      = [-1100,1100]
dBNorm_limits = [10**-15,10**-1]

dBx_sum_limits = [-0.4,0.4]
dBy_sum_limits = [-0.4,0.4]
dBz_sum_limits = [-50,50]
dB_sum_limits  = [0,50]
diff_limits    = [0,0.2]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import exists
from os import makedirs
import swmfio
from copy import deepcopy
import logging

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.8,7.2]
# plt.rcParams["figure.figsize"] = [3.6,3.2]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 7
plt.rcParams['axes.grid'] = True

def nez(time, pos, csys):
  """Unit vectors in geographic north, east, and zenith dirs"""

  from hxform import hxform as hx
  import numpy as np

  # z axis in geographic
  Z = hx.transform(np.array([0, 0, 1]), time, 'GEO', csys, lib='cxform')

  # zenith direction ("up")
  z_geo = pos/np.linalg.norm(pos)

  e_geo = np.cross(z_geo, Z)
  e_geo = e_geo/np.linalg.norm(e_geo)

  n_geo = np.cross(e_geo, z_geo)
  n_geo = n_geo/np.linalg.norm(n_geo)

  return n_geo, e_geo, z_geo

def create_directory( folder ):
    """ If directory for output files does not exist, create it
    
    Inputs:
        folder = basename of folder.  Complete path to folder is:
            target + folder
    Outputs:
        None 
     """
     
    logging.info( 'Looking for directory: ' + target + folder )
    if not exists( target + folder ): 
        logging.info( 'Creating directory: ' + target + folder )
        makedirs( target + folder )
    return 

def plot_db_Norm_r( r, dBmag, dBmagNorm, dBxNorm, dByNorm, dBzNorm, title, base ):
    """Plot various forms of the magnitude of dB in each cell versus radius r
    
    Inputs:
        r = radius from earth's center in units of Re
        dBmag, dBmagNorm = magnitude of dB in each cell.  dBmagNorm is normalized.
            (Cells on the edges of the grid are much larger than cells near earth, 
             so we normalize them to the smallest cell.)
        dBxNorm, dByNorm, dBzNorm = x,y,z components of dB in each cell, normalized
            to the smallest cell.  (Cells on the edges of the grid are much larger
            than cells near earth, so we normalize them to the smallest cell.)
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
         
    # Plot dB mag (normalized by cell volume) as a function of range r
    plt.subplot(2,4,1).scatter(x=r, y=dBxNorm, s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_x |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15,10**-1)
 
    plt.subplot(2,4,2).scatter(x=r, y=dByNorm, s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_y |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15,10**-1)
 
    plt.subplot(2,4,3).scatter(x=r, y=dBzNorm, s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_z |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15,10**-1)
 
    plt.subplot(2,4,4).scatter(x=r, y=dBmagNorm, s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15,10**-1)
 
    fig = plt.gcf()
    create_directory( 'png-combined-dBNorm-r/' ) 
    logging.info(f'Saving {base} combined dBNorm plot')
    fig.savefig( target + 'png-combined-dBNorm-r/' + base + '.out.combined-dBNorm-r.png')
    plt.close(fig)

    return

def plot_cumulative_B( r, df_r, title, base ):
    """Plot various forms of the cumulative sum of dB in each cell versus 
        range r.  To generate the cumulative sum, we order the cells in terms of
        range r from the earth's center.  We start with a small sphere and 
        vector sum all of the dB contributions inside the sphere.  Expand the
        sphere slightly and resum.  Repeat until all cells are in the sum.
        
    Inputs:
        r = radius from earth's center in units of Re
        df_r = dataframe containing cumulative sums ordered from small r to 
            large r
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
        
    # Plot the cummulative sum of dB as a function of r   
    df_plt1 = df_r.plot.scatter(x='r', y='dBxSum', 
                              ax = plt.subplot(2,4,1),
                              xlim = [1,1000], 
                              ylim = dBx_sum_limits,
                              logx = True,  
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_x $')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt2 = df_r.plot.scatter(x='r', y='dBySum', 
                              ax = plt.subplot(2,4,2),
                              xlim = [1,1000], 
                              ylim = dBy_sum_limits,
                              logx = True, 
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_y$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt3 = df_r.plot.scatter(x='r', y='dBzSum', 
                              ax = plt.subplot(2,4,3),
                              xlim = [1,1000], 
                              ylim = dBz_sum_limits,
                              logx = True, 
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_z$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt4 = df_r.plot.scatter(x='r', y='dBSumMag', 
                              ax = plt.subplot(2,4,4),
                              xlim = [1,1000], 
                              ylim = dB_sum_limits,
                              logx = True,
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$| \Sigma_r \delta B |$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt5 = df_r.plot.scatter(x='r', y='diff', 
                              ax = plt.subplot(2,4,8),
                              xlim = [1,1000], 
                              ylim = diff_limits,
                              logx = True,
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$| \Sigma_r \delta B | - | \Sigma_r \delta B_z |$')
  
    fig = plt.gcf()
    create_directory( 'png-combined-sum-dB-r/' ) 
    logging.info(f'Saving {base} combined dB plot')
    fig.savefig( target + 'png-combined-sum-dB-r/' + base + '.out.combined-sum-dB-r.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5

    return

def plot_cumulative_B_cutoff( r, df, title, base ):
    """Plot the cumulative sum of dB in each cell versus range r.  Only
        include values of dB with |j| > Threshold.  To generate the cumulative 
        sum, we order the cells in terms of range r from the earth's center.  
        We start with a small sphere and vector sum all of the dB contributions 
        inside the sphere.  Expand the sphere slightly and resum.  Repeat until 
        all cells are in the sum.
        
    Inputs:
        r = radius from earth's center in units of Re
        df = dataframe containing dB data
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
    
    df_r = df.sort_values(by='r', ascending=True)
    thres = df_r['jMag'].max() / 1000.0
    df_r = df_r.drop(df_r[df_r['jMag'] < thres].index)
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBxSumMag'] = df_r['dBxSum'].abs()
    df_r['dBySumMag'] = df_r['dBySum'].abs()
    df_r['dBzSumMag'] = df_r['dBzSum'].abs()
    df_r['dBSumMag'] = (df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)**(1/2)
    df_r['diff'] = df_r['dBSumMag'] - df_r['dBzSumMag']

        
    # Plot the cummulative sum of dB as a function of r   
    df_plt1 = df_r.plot.scatter(x='r', y='dBxSum', 
                              ax = plt.subplot(2,4,1),
                              xlim = [1,1000], 
                              ylim = dBx_sum_limits,
                              logx = True,  
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_x $')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt2 = df_r.plot.scatter(x='r', y='dBySum', 
                              ax = plt.subplot(2,4,2),
                              xlim = [1,1000], 
                              ylim = dBy_sum_limits,
                              logx = True, 
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_y$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt3 = df_r.plot.scatter(x='r', y='dBzSum', 
                              ax = plt.subplot(2,4,3),
                              xlim = [1,1000], 
                              ylim = dBz_sum_limits,
                              logx = True, 
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$\Sigma_r \delta B_z$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt4 = df_r.plot.scatter(x='r', y='dBSumMag', 
                              ax = plt.subplot(2,4,4),
                              xlim = [1,1000], 
                              ylim = dB_sum_limits,
                              logx = True,
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$| \Sigma_r \delta B |$')
  
    # Plot the cummulative sum of dB as a function of r   
    df_plt5 = df_r.plot.scatter(x='r', y='diff', 
                              ax = plt.subplot(2,4,8),
                              xlim = [1,1000], 
                              ylim = diff_limits,
                              logx = True,
                              title = title, 
                              s=1, 
                              xlabel='$r/R_E$', 
                              ylabel=r'$| \Sigma_r \delta B | - | \Sigma_r \delta B_z |$')
  
    fig = plt.gcf()
    create_directory( 'png-combined-sum-dB-cutoff-r/' ) 
    logging.info(f'Saving {base} combined dB with cutoff plot')
    fig.savefig( target + 'png-combined-sum-dB-cutoff-r/' + base + '.out.combined-sum-dB-cutoff-r.png')
    plt.close(fig)
    
    del df_r
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5

    return


def plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base ):
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
    
    # Plot rho as a function of range r
    df_plt1 = df_day.plot.scatter(x='r', y='rho', 
                                ax = plt.subplot(2,4,1),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=rho_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$\rho$', 
                                title='Day ' + title, 
                                s=1)
    
    df_plt2 = df_night.plot.scatter(x='r', y='rho', 
                                ax = plt.subplot(2,4,5),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=rho_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$\rho$', 
                                title='Night ' + title, 
                                s=1)

    # Plot p as a function of range r
    df_plt3 = df_day.plot.scatter(x='r', y='p', 
                                ax = plt.subplot(2,4,2),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=p_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$p$', 
                                title='Day ' + title, 
                                s=1)

    df_plt4 = df_night.plot.scatter(x='r', y='p', 
                                ax = plt.subplot(2,4,6),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=p_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$p$', 
                                title='Night ' + title, 
                                s=1)

    # Plot jMag as a function of range r
    df_plt5 = df_day.plot.scatter(x='r', y='jMag', 
                                ax = plt.subplot(2,4,3),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=jMag_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|j|$', 
                                title='Day ' + title, 
                                s=1)

    df_plt6 = df_night.plot.scatter(x='r', y='jMag', 
                                ax = plt.subplot(2,4,7),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=jMag_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|j|$', 
                                title='Night ' + title, 
                                s=1)

    # Plot uMag as a function of range r
    df_plt7 = df_day.plot.scatter(x='r', y='uMag', 
                                ax = plt.subplot(2,4,4),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=uMag_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|u|$', 
                                title='Day ' + title, 
                                s=1)

    df_plt8 = df_night.plot.scatter(x='r', y='uMag', 
                                ax = plt.subplot(2,4,8),
                                logx=True, 
                                logy=True, 
                                xlim=[1,10**3], 
                                ylim=uMag_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|u|$', 
                                title='Night ' + title, 
                                s=1)

    fig = plt.gcf()
    create_directory( 'png-combined-day-night/' ) 
    logging.info(f'Saving {base} combined day-night plot')
    fig.savefig( target + 'png-combined-day-night/' + base + '.out.png-combined-day-night.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return

def plot_jx_jy_jz_day_night( df_day, df_night, title, base ):
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
    
    # Plot jx as a function of range r
    df_plt1 = df_day.plot.scatter(x='r', y='jx', 
                                ax = plt.subplot(2,4,1),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_x$', 
                                title='Day ' + title, 
                                s=1)
 
    df_plt2 = df_night.plot.scatter(x='r', y='jx', 
                                ax = plt.subplot(2,4,5),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_x$', 
                                title='Night ' + title, 
                                s=1)

    # Plot jy as a function of range r
    df_plt3 = df_day.plot.scatter(x='r', y='jy', 
                                ax = plt.subplot(2,4,2),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_y$', 
                                title='Day ' + title, 
                                s=1)

    df_plt4 = df_night.plot.scatter(x='r', y='jy', 
                                ax = plt.subplot(2,4,6),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_y$', 
                                title='Night ' + title, 
                                s=1)

    # Plot jz as a function of range r
    df_plt5 = df_day.plot.scatter(x='r', y='jz', 
                                ax = plt.subplot(2,4,3),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_z$', 
                                title='Day ' + title, 
                                s=1)

    df_plt6 = df_night.plot.scatter(x='r', y='jz', 
                                ax = plt.subplot(2,4,7),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$j_z$', 
                                title='Night ' + title, 
                                s=1)

    # Plot jMag as a function of range r
    df_plt7 = df_day.plot.scatter(x='r', y='jMag', 
                                ax = plt.subplot(2,4,4),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|j|$', 
                                title='Day ' + title, 
                                s=1)

    df_plt8 = df_night.plot.scatter(x='r', y='jMag', 
                                ax = plt.subplot(2,4,8),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=j_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|j|$', 
                                title='Night ' + title, 
                                s=1)

    fig = plt.gcf()
    create_directory( 'png-combined-jxyz-day-night/' ) 
    logging.info(f'Saving {base} combined jx, jy, jz day-night plot')
    fig.savefig( target + 'png-combined-jxyz-day-night/' + base + '.out.png-combined-jxyz-day-night.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return

def plot_jx_jy_jz_vs_x( df, title, base ):
    """Plot jx, jy, jz  in each cell versus x.  
        
    Inputs:
        df = dataframe containing jx, jy, jz and x
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
    
    # Plot jx as a function of range x
    df_plt1 = df.plot.scatter(x='x', y='jx', 
                                ax = plt.subplot(2,4,1),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_x$', 
                                title=title, 
                                s=1)
 
    df_plt2 = df.plot.scatter(x='x', y='jx', 
                                ax = plt.subplot(2,4,5),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_x$', 
                                title=title, 
                                s=1)
 
    # Plot jy as a function of range r
    df_plt3 = df.plot.scatter(x='x', y='jy', 
                                ax = plt.subplot(2,4,2),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_y$', 
                                title=title, 
                                s=1)

    df_plt4 = df.plot.scatter(x='x', y='jy', 
                                ax = plt.subplot(2,4,6),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_y$', 
                                title=title, 
                                s=1)

    # Plot jz as a function of range x
    df_plt5 = df.plot.scatter(x='x', y='jz', 
                                ax = plt.subplot(2,4,3),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_z$', 
                                title=title, 
                                s=1)

    df_plt6 = df.plot.scatter(x='x', y='jz', 
                                ax = plt.subplot(2,4,7),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$j_z$', 
                                title=title, 
                                s=1)

    # Plot jMag and r as a function of range x
    df_plt7 = df.plot.scatter(x='x', y='jMag', 
                                ax = plt.subplot(2,4,4),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ x/R_E $', 
                                ylabel=r'$|j|$', 
                                title=title, 
                                s=1)
    
    df_plt8 = df.plot.scatter(x='x', y='r', 
                               ax = plt.subplot(2,4,8),
                               xlim=[-300,300], 
                               ylim=[-300,300], 
                               xlabel=r'$ x/R_E $', 
                               ylabel=r'$r/R_E$', 
                               title=title, 
                               s=1)


    fig = plt.gcf()
    create_directory( 'png-combined-jxyz-x/' ) 
    logging.info(f'Saving {base} combined jx, jy, jz vs x plot')
    fig.savefig( target + 'png-combined-jxyz-x/' + base + '.out.png-combined-jxyz-x.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return

def plot_jr_jt_jp_vs_x( df, title, base, coord = 'x' ):
    """Plot jr, jtheta, jphi  in each cell versus x.  
        
    Inputs:
        df = dataframe containing jr, jtheta, jphi and x
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """
    
    # Plot jr as a function of range x
    df_plt1 = df.plot.scatter(x=coord, y='jr', 
                                ax = plt.subplot(2,4,1),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_r$', 
                                title=title, 
                                s=1)
 
    df_plt2 = df.plot.scatter(x=coord, y='jr', 
                                ax = plt.subplot(2,4,5),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_r$', 
                                title=title, 
                                s=1)
 
    # Plot jtheta as a function of range r
    df_plt3 = df.plot.scatter(x=coord, y='jtheta', 
                                ax = plt.subplot(2,4,2),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_\theta$', 
                                title=title, 
                                s=1)

    df_plt4 = df.plot.scatter(x=coord, y='jtheta', 
                                ax = plt.subplot(2,4,6),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_\theta$', 
                                title=title, 
                                s=1)

    # Plot jphi as a function of range x
    df_plt5 = df.plot.scatter(x=coord, y='jphi', 
                                ax = plt.subplot(2,4,3),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_\phi$', 
                                title=title, 
                                s=1)

    df_plt6 = df.plot.scatter(x=coord, y='jphi', 
                                ax = plt.subplot(2,4,7),
                                xlim=[-15,15], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$j_\phi$', 
                                title=title, 
                                s=1)

    # Plot jMag and r as a function of range x
    df_plt7 = df.plot.scatter(x=coord, y='jMag', 
                                ax = plt.subplot(2,4,4),
                                xlim=[-300,300], 
                                ylim=j_limits,
                                xlabel=r'$ ' + coord + '/R_E $', 
                                ylabel=r'$|j|$', 
                                title=title, 
                                s=1)
    
    df_plt8 = df.plot.scatter(x='jMag', y='jMag2', 
                               ax = plt.subplot(2,4,8),
                               xlim=[0,j_limits[1]], 
                               ylim=[0,j_limits[1]], 
                               xlabel=r'$|j(x,y,z)|$', 
                               ylabel=r'$|j(r,\theta,\phi)|$', 
                               title=title, 
                               s=1)


    fig = plt.gcf()
    create_directory( 'png-combined-jrtp-'+coord+'/' ) 
    logging.info(f'Saving {base} combined jr, jtheta, jphi vs ' + coord + ' plot')
    fig.savefig( target + 'png-combined-jrtp-'+coord+'/'  + base + '.out.'+'png-combined-jrtp-'+coord+'.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return


def plot_ux_uy_uz_day_night( df_day, df_night, title, base ):
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
    
    # Plot ux as a function of range r
    df_plt1 = df_day.plot.scatter(x='r', y='ux', 
                                ax = plt.subplot(2,4,1),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_x$', 
                                title='Day ' + title, 
                                s=1)
 
    df_plt2 = df_night.plot.scatter(x='r', y='ux', 
                                ax = plt.subplot(2,4,5),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_x$', 
                                title='Night ' + title, 
                                s=1)

    # Plot uy as a function of range r
    df_plt3 = df_day.plot.scatter(x='r', y='uy', 
                                ax = plt.subplot(2,4,2),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_y$', 
                                title='Day ' + title, 
                                s=1)

    df_plt4 = df_night.plot.scatter(x='r', y='uy', 
                                ax = plt.subplot(2,4,6),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_y$', 
                                title='Night ' + title, 
                                s=1)

    # Plot uz as a function of range r
    df_plt5 = df_day.plot.scatter(x='r', y='uz', 
                                ax = plt.subplot(2,4,3),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_z$', 
                                title='Day ' + title, 
                                s=1)

    df_plt6 = df_night.plot.scatter(x='r', y='uz', 
                                ax = plt.subplot(2,4,7),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$u_z$', 
                                title='Night ' + title, 
                                s=1)

    # Plot uMag as a function of range r
    df_plt7 = df_day.plot.scatter(x='r', y='uMag', 
                                ax = plt.subplot(2,4,4),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|u|$', 
                                title='Day ' + title, 
                                s=1)

    df_plt8 = df_night.plot.scatter(x='r', y='uMag', 
                                ax = plt.subplot(2,4,8),
                                logx=True, 
                                xlim=[1,10**3], 
                                ylim=u_limits,
                                xlabel=r'$ r/R_E $', 
                                ylabel=r'$|u|$', 
                                title='Night ' + title, 
                                s=1)

    fig = plt.gcf()
    create_directory( 'png-combined-uxyz-day-night/' ) 
    logging.info(f'Saving {base} combined ux, uy, uz day-night plot')
    fig.savefig( target + 'png-combined-uxyz-day-night/' + base + '.out.png-combined-uxyz-day-night.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return


def plot_dBnorm_day_night( df_day, df_night, title, base ):
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
    
    # Plot dBmagNorm as a function of rho
    df_plt1 = df_day.plot.scatter(x='rho', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,1),
                                 logx=True, 
                                 logy=True, 
                                 xlim=rho_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$\rho$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Day ' + title, 
                                 s=1)

    df_plt2 = df_night.plot.scatter(x='rho', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,5),
                                 logx=True, 
                                 logy=True, 
                                 xlim=rho_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$\rho$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Night ' + title, 
                                 s=1)

    # Plot dBmagNorm as a function of p
    df_plt3 = df_day.plot.scatter(x='p', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,2),
                                 logx=True, 
                                 logy=True, 
                                 xlim=p_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$p$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Day ' + title, 
                                 s=1)

    df_plt4 = df_night.plot.scatter(x='p', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,6),
                                 logx=True, 
                                 logy=True, 
                                 xlim=p_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$p$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Night ' + title, 
                                 s=1)

    # Plot dBmagNorm as a function of jMag
    df_plt5 = df_day.plot.scatter(x='jMag', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,3),
                                 logx=True, 
                                 logy=True, 
                                 xlim=jMag_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$| j |$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Day ' + title, 
                                 s=1)

    df_plt6 = df_night.plot.scatter(x='jMag', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,7),
                                 logx=True, 
                                 logy=True, 
                                 xlim=jMag_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$| j |$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Night ' + title, 
                                 s=1)

    # Plot dBmagNorm as a function of uMag
    df_plt7 = df_day.plot.scatter(x='uMag', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,4),
                                 logx=True, 
                                 logy=True, 
                                 xlim=uMag_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$| u |$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Day ' + title, 
                                 s=1)

    df_plt8 = df_night.plot.scatter(x='uMag', y='dBmagNorm', 
                                 ax = plt.subplot(2,4,8),
                                 logx=True, 
                                 logy=True, 
                                 xlim=uMag_limits, 
                                 ylim=dBNorm_limits,
                                 xlabel=r'$| u |$', 
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)', 
                                 title='Night ' + title, 
                                 s=1)

    fig = plt.gcf()
    create_directory( 'png-combined-dBNorm-day-night/' ) 
    logging.info(f'Saving {base} combined dBNorm day-night plot')
    fig.savefig( target + 'png-combined-dBNorm-day-night/' + base + '.out.png-combined-dBNorm-day-night.png')
    plt.close(fig)
    
    del df_plt1
    del df_plt2
    del df_plt3
    del df_plt4
    del df_plt5
    del df_plt6
    del df_plt7
    del df_plt8

    return


def process_data( base, dirpath = origin ):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
        
    Inputs:
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
        title = title to use in plots, which is derived from base (file basename)
        batsrus = BATSRUS data read by swmfio 
    """
           
    # Get location of Colaba
    # from hxform import hxform as hx
    # import numpy as np
    
    # time = '2001-09-02T04:10:00' # Should be 2019
    # pos = (1., 18.907, 72.815)   # Geographic r, lat, long of Colaba
    # from hxform import hxform as hx
    # pos = hx.transform(np.array(pos), time, 'GEO', 'GSM', ctype_in="sph", ctype_out="car", lib='cxform')
    
    # n_geo, e_geo, z_geo = nez(time, pos, "GSM")
    # (X,Y,Z) = n_geo # Point to calculate delta B (dB)

    logging.info('Parsing BATSRUS file...')
    
    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')
    
    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert( batsrus != None )
       
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    x = batsrus.data_arr[:,var_dict['x']][:]
    y = batsrus.data_arr[:,var_dict['y']][:]
    z = batsrus.data_arr[:,var_dict['z']][:]
        
    bx = batsrus.data_arr[:,var_dict['bx']][:]
    by = batsrus.data_arr[:,var_dict['by']][:]
    bz = batsrus.data_arr[:,var_dict['bz']][:]
        
    jx = batsrus.data_arr[:,var_dict['jx']][:]
    jy = batsrus.data_arr[:,var_dict['jy']][:]
    jz = batsrus.data_arr[:,var_dict['jz']][:]
        
    ux = batsrus.data_arr[:,var_dict['ux']][:]
    uy = batsrus.data_arr[:,var_dict['uy']][:]
    uz = batsrus.data_arr[:,var_dict['uz']][:]
        
    p = batsrus.data_arr[:,var_dict['p']][:]
    rho = batsrus.data_arr[:,var_dict['rho']][:]
    measure = batsrus.data_arr[:,var_dict['measure']][:]

    # Get the smallest cell (by volume), we will use it to normalize all
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize everthing
    # to the smallest cell.
    minMeasure = np.amin(measure)
    
    logging.info('Calculating delta B...')
    
    # Calculate new quantities
    r = ((X-x)**2+(Y-y)**2+(Z-z)**2)**(1/2)
    factor = 637.1*measure/r**3
    
    dBx = factor*( jy*(Z-z) - jz*(Y-y) )
    dBy = factor*( jz*(X-x) - jx*(Z-z) )
    dBz = factor*( jx*(Y-y) - jy*(X-x) )

    dBx[r < rCurrents] = 0.0
    dBy[r < rCurrents] = 0.0
    dBz[r < rCurrents] = 0.0

    dBmag = (dBx**2 + dBy**2 + dBz**2)**(1/2)
    dBmagNorm = dBmag * minMeasure/measure
    dBxNorm = np.abs(dBx * minMeasure/measure)
    dByNorm = np.abs(dBy * minMeasure/measure)
    dBzNorm = np.abs(dBz * minMeasure/measure)

    jMag = np.sqrt( jx**2 + jy**2 + jz**2 )
    uMag = np.sqrt( ux**2 + uy**2 + uz**2 )
 
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1]

    logging.info('Creating cumulative sum dB dataframe...')
     
    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.
    df1 = pd.DataFrame()
    df1['dBx'] = deepcopy(dBx)
    df1['dBy'] = deepcopy(dBy)
    df1['dBz'] = deepcopy(dBz)
    df1['r'] = deepcopy(r)
    df1['jMag'] = deepcopy(jMag)
    
    df_r = df1.sort_values(by='r', ascending=True)
    
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBxSumMag'] = df_r['dBxSum'].abs()
    df_r['dBySumMag'] = df_r['dBySum'].abs()
    df_r['dBzSumMag'] = df_r['dBzSum'].abs()
    df_r['dBSumMag'] = (df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)**(1/2)
    df_r['diff'] = df_r['dBSumMag'] - df_r['dBzSumMag']
   
    logging.info('Creating jr, jtheta, jphi, dayside/nightside dataframe...')
    
    # Make a new dataframe that we use for spherical coordinate transform
    # and separating the data into dayside and nightside regions
    
    df2 = pd.DataFrame()
    df2['x'] = deepcopy(x)
    df2['y'] = deepcopy(y)
    df2['z'] = deepcopy(z)
    df2['rho'] = deepcopy(rho)
    df2['p'] = deepcopy(p)
    df2['jMag'] = deepcopy(jMag)
    df2['jx'] = deepcopy(jx)
    df2['jy'] = deepcopy(jy)
    df2['jz'] = deepcopy(jz)
    df2['uMag'] = deepcopy(uMag)
    df2['ux'] = deepcopy(ux)
    df2['uy'] = deepcopy(uy)
    df2['uz'] = deepcopy(uz)
    df2['r'] = deepcopy(r)
    df2['dBmagNorm'] = deepcopy(dBmagNorm)
    
    #################################
    #################################
    # thres = df2['jMag'].max() / 10.0
    # df2 = df2.drop(df2[df2['jMag'] < thres].index)
    #################################
    #################################

    # Transform the currents, j, into spherical coordinates
    
    # Determine theta and phi of the radius vector from the origin to the 
    # center of the cell
    df2['theta'] = np.arccos( df2['z']/df2['r'] )
    df2['phi'] = np.arctan2( df2['y'], df2['x'] )
    
    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df2['jr'] = df2['jx'] * np.sin(df2['theta']) * np.cos(df2['phi']) + \
        df2['jy'] * np.sin(df2['theta']) * np.sin(df2['phi']) + \
        df2['jz'] * np.cos(df2['theta'])
        
    df2['jtheta'] = df2['jx'] * np.cos(df2['theta']) * np.cos(df2['phi']) + \
        df2['jy'] * np.cos(df2['theta']) * np.sin(df2['phi']) - \
        df2['jz'] * np.sin(df2['theta'])
        
    df2['jphi'] = - df2['jx'] * np.sin(df2['phi']) + df2['jy'] * np.cos(df2['phi'])
    
    # Recalculate the magnitude jMag2, which should be equal to jMag.
    # Use as an idiot check of calculation
    df2['jMag2'] = (df2['jr']**2 + df2['jtheta']**2 + df2['jphi']**2)**(1/2)

    # #################################
    # #################################    
    # # Drop asymmetric jr vs y lobes
    # df2 = df2.drop( df2[np.logical_and(df2['jr'].abs() > 0.007, df2['y'].abs() < 4)].index)
    # #################################
    # #################################
    # # Drop jphi vs y blob
    # df2 = df2.drop( df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
    # #################################
    # #################################
    # # Drop jphi vs z blob
    # df2 = df2.drop( df2[np.logical_and(df2['jphi'].abs() > 0.007, df2['z'].abs() < 2)].index)
    # #################################
    # #################################

    # Split the data into dayside (x>=0) and nightside (x<0)
    df_tmp = df2[df2['r'] > rCurrents]
    # df_day = df_tmp[df_tmp['x'] >= 0]
    # df_night = df_tmp[df_tmp['x'] < 0]

    # Do plots...

    # logging.info('Creating dB (Norm) vs r plots...')
    # plot_db_Norm_r( r, dBmag, dBmagNorm, dBxNorm, dByNorm, dBzNorm, title, base )

    # logging.info('Creating cumulative Sum B vs r plots...')
    # plot_cumulative_B( r, df_r, title, base )
    
    # logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    # plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating day/night jx, jy, jz vs r plots...')
    # plot_jx_jy_jz_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating jx, jy, jz vs x plots...')
    # plot_jx_jy_jz_vs_x( df_tmp, title, base)
    
    # logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    # plot_jr_jt_jp_vs_x( df_tmp, title, base, coord = 'x')
    # plot_jr_jt_jp_vs_x( df_tmp, title, base, coord = 'y')
    # plot_jr_jt_jp_vs_x( df_tmp, title, base, coord = 'z')

    # logging.info('Creating day/night ux, uy, uz vs r plots...')
    # plot_ux_uy_uz_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    # plot_dBnorm_day_night( df_day, df_night, title, base )
    
    # logging.info('Creating cumulative Sum B with j cutoff vs r plots...')
    # plot_cumulative_B_cutoff( r, df1, title, base )
    
    # Get rid of dataframes that are no longer needed
    del df_r
    del df1
    del df2
    del df_tmp
    # del df_day
    # del df_night

    # Get rid of other data that is no longer needed
    del batsrus
    
    return
   
def process_data_dbl_chk( base, dirpath = origin ):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
    Use the data to run a double-check on the results.  We will do the cumulative
    sums in multiple ways
        
    Inputs:
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        None
    """
           
    logging.info('Parsing BATSRUS file...')
    
    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')
    
    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert( batsrus != None )
       
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    x = batsrus.data_arr[:,var_dict['x']][:]
    y = batsrus.data_arr[:,var_dict['y']][:]
    z = batsrus.data_arr[:,var_dict['z']][:]
        
    jx = batsrus.data_arr[:,var_dict['jx']][:]
    jy = batsrus.data_arr[:,var_dict['jy']][:]
    jz = batsrus.data_arr[:,var_dict['jz']][:]
        
    measure = batsrus.data_arr[:,var_dict['measure']][:]
    
    logging.info('Calculating delta B for double check...')
    
    # Calculate new quantities
    r = ((X-x)**2+(Y-y)**2+(Z-z)**2)**(1/2)
    factor = 637.1*measure/r**3
    
    dBx = factor*( jy*(Z-z) - jz*(Y-y) )
    dBy = factor*( jz*(X-x) - jx*(Z-z) )
    dBz = factor*( jx*(Y-y) - jy*(X-x) )

    dBx[r < rCurrents] = 0.0
    dBy[r < rCurrents] = 0.0
    dBz[r < rCurrents] = 0.0

    ########################
    ########################
    # We see difference if dBcutoff smaller than 0.01, e.g., 0.001
    # But for dBcutoff = 0.01, all the fields are zero.
    ########################
    ########################
    # dBcutoff = 0.001
    # dBx[dBx < dBcutoff] = 0.0
    # dBy[dBy < dBcutoff] = 0.0
    # dBz[dBz < dBcutoff] = 0.0

    # Use numpy cumulative sum to be used in double check
    logging.info('Numpy cumulative sum of dB for double check...')

    BxSum2 = np.cumsum(dBx)
    BySum2 = np.cumsum(dBy)
    BzSum2 = np.cumsum(dBz)
    
    Bxtot2 = BxSum2[-1]
    Bytot2 = BySum2[-1]
    Bztot2 = BzSum2[-1]
        
    Btot2 = (Bxtot2**2 + Bytot2**2 + Bztot2**2)**(1/2)

    logging.info('Creating cumulative sum dB dataframe for double check...')
     
    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.
    df1 = pd.DataFrame()
    df1['dBx'] = deepcopy(dBx)
    df1['dBy'] = deepcopy(dBy)
    df1['dBz'] = deepcopy(dBz)
    df1['r'] = deepcopy(r)
    
    df1 = df1.sort_values(by='r', ascending=True)

    df1['dBxSum'] = df1['dBx'].cumsum()
    df1['dBySum'] = df1['dBy'].cumsum()
    df1['dBzSum'] = df1['dBz'].cumsum()
    df1['dBSumMag'] = (df1['dBxSum']**2 + df1['dBySum']**2 + df1['dBzSum']**2)**(1/2)
   
    Btot1 = df1['dBSumMag'].iloc[-1]
    Bxtot1 = df1['dBxSum'].iloc[-1]
    Bytot1 = df1['dBySum'].iloc[-1]
    Bztot1 = df1['dBzSum'].iloc[-1]
    
    # Do a brute force cumulative sum for double check
    logging.info('Brute force (loop) cumulative sum of dB for double check...')

    Bxtot3 = 0
    Bytot3 = 0
    Bztot3 = 0
    
    for i in range( len(dBx) ):
        Bxtot3 = Bxtot3 + dBx[i]
        Bytot3 = Bytot3 + dBy[i]
        Bztot3 = Bztot3 + dBz[i]
    
    Btot3 = (Bxtot3**2 + Bytot3**2 + Bztot3**2)**(1/2)
    
    # Calculate differences between approaches
    Bxdiff1 = Bxtot1 - Bxtot2
    Bydiff1 = Bytot1 - Bytot2
    Bzdiff1 = Bztot1 - Bztot2
    Bdiff1 = Btot1 - Btot2
    
    Bxdiff2 = Bxtot2 - Bxtot3
    Bydiff2 = Bytot2 - Bytot3
    Bzdiff2 = Bztot2 - Bztot3
    Bdiff2 = Btot2 - Btot3
    
    Bxdiff3 = Bxtot1 - Bxtot3
    Bydiff3 = Bytot1 - Bytot3
    Bzdiff3 = Bztot1 - Bztot3
    Bdiff3 = Btot1 - Btot3
    
    ############################
    ############################
    #
    # I have a concern, all three methods of calculating the cumulative
    # sum give slightly different results.
    #
    ############################
    ############################
        
    logging.info(f'{base} magnitude diff Bx = {Bxdiff1}')
    logging.info(f'{base} magnitude diff Bx = {Bxdiff2}')
    logging.info(f'{base} magnitude diff Bx = {Bxdiff3}')
    logging.info(f'{base} magnitude diff By = {Bydiff1}')
    logging.info(f'{base} magnitude diff By = {Bydiff2}')
    logging.info(f'{base} magnitude diff By = {Bydiff3}')
    logging.info(f'{base} magnitude diff Bz = {Bzdiff1}')
    logging.info(f'{base} magnitude diff Bz = {Bzdiff2}')
    logging.info(f'{base} magnitude diff Bz = {Bzdiff3}')
    logging.info(f'{base} magnitude diff B = {Bdiff1}')
    logging.info(f'{base} magnitude diff B = {Bdiff2}')
    logging.info(f'{base} magnitude diff B = {Bdiff3}')
    
    return 

def process_data_compare_weigel( base = '3d__mhd_4_e20100320-000000-000', dirpath = origin ):
    """Process data in BATSRUS file to create dataframe with calculated quantities.
        
    Inputs:
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        df = dataframe containing data from vtk file plus additional calculated
            parameters
        title = title to use in plots, which is derived from base (file basename)
        batsrus = BATSRUS data read by swmfio 
    """

    from math import isclose
    
    logging.info('Parsing BATSRUS file...')
    
    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')
    
    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert( batsrus != None )
       
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    x = batsrus.data_arr[:,var_dict['x']][:]
    y = batsrus.data_arr[:,var_dict['y']][:]
    z = batsrus.data_arr[:,var_dict['z']][:]
        
    bx = batsrus.data_arr[:,var_dict['bx']][:]
    by = batsrus.data_arr[:,var_dict['by']][:]
    bz = batsrus.data_arr[:,var_dict['bz']][:]
        
    jx = batsrus.data_arr[:,var_dict['jx']][:]
    jy = batsrus.data_arr[:,var_dict['jy']][:]
    jz = batsrus.data_arr[:,var_dict['jz']][:]
        
    ux = batsrus.data_arr[:,var_dict['ux']][:]
    uy = batsrus.data_arr[:,var_dict['uy']][:]
    uz = batsrus.data_arr[:,var_dict['uz']][:]
        
    p = batsrus.data_arr[:,var_dict['p']][:]
    rho = batsrus.data_arr[:,var_dict['rho']][:]
    measure = batsrus.data_arr[:,var_dict['measure']][:]

    # Get the smallest cell (by volume), we will use it to normalize all
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize everthing
    # to the smallest cell.
    minMeasure = np.amin(measure)
    
    logging.info('Calculating delta B...')
    
    # Calculate new quantities
    r = ((X-x)**2+(Y-y)**2+(Z-z)**2)**(1/2)
    factor = 637.1*measure/r**3
    
    dBx = factor*( jy*(Z-z) - jz*(Y-y) )
    dBy = factor*( jz*(X-x) - jx*(Z-z) )
    dBz = factor*( jx*(Y-y) - jy*(X-x) )

    dBx[r < rCurrents] = 0.0
    dBy[r < rCurrents] = 0.0
    dBz[r < rCurrents] = 0.0

    dBmag = (dBx**2 + dBy**2 + dBz**2)**(1/2)
    dBmagNorm = dBmag * minMeasure/measure
    dBxNorm = np.abs(dBx * minMeasure/measure)
    dByNorm = np.abs(dBy * minMeasure/measure)
    dBzNorm = np.abs(dBz * minMeasure/measure)

    jMag = np.sqrt( jx**2 + jy**2 + jz**2 )
    uMag = np.sqrt( ux**2 + uy**2 + uz**2 )
 
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1]

    logging.info('Creating dB dataframe...')
     
    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.
    df1 = pd.DataFrame()
    
    df1['x'] = deepcopy(x)
    df1['y'] = deepcopy(y)
    df1['z'] = deepcopy(z)
    df1['r'] = deepcopy(r)
        
    df1['dBx'] = deepcopy(dBx)
    df1['dBy'] = deepcopy(dBy)
    df1['dBz'] = deepcopy(dBz)
    df1['dBmag'] = deepcopy(dBmag)
    
    df1['bx'] = deepcopy(ux)
    df1['by'] = deepcopy(uy)
    df1['bz'] = deepcopy(uz)
 
    df1['jMag'] = deepcopy(jMag)
    df1['jx'] = deepcopy(jx)
    df1['jy'] = deepcopy(jy)
    df1['jz'] = deepcopy(jz)
    
    df1['uMag'] = deepcopy(uMag)
    df1['ux'] = deepcopy(ux)
    df1['uy'] = deepcopy(uy)
    df1['uz'] = deepcopy(uz)
     
    df1['rho'] = deepcopy(rho)
    df1['p'] = deepcopy(p)
    df1['measure'] = deepcopy(measure)
   
    # Determine theta and phi of the radius vector from the origin to the 
    # center of the cell
    df1['theta'] = np.arccos( df1['z']/df1['r'] )
    df1['phi'] = np.arctan2( df1['y'], df1['x'] )
    
    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df1['jr'] = df1['jx'] * np.sin(df1['theta']) * np.cos(df1['phi']) + \
        df1['jy'] * np.sin(df1['theta']) * np.sin(df1['phi']) + \
        df1['jz'] * np.cos(df1['theta'])
        
    df1['jtheta'] = df1['jx'] * np.cos(df1['theta']) * np.cos(df1['phi']) + \
        df1['jy'] * np.cos(df1['theta']) * np.sin(df1['phi']) - \
        df1['jz'] * np.sin(df1['theta'])
        
    df1['jphi'] = - df1['jx'] * np.sin(df1['phi']) + df1['jy'] * np.cos(df1['phi'])
    
    logging.info('Parsing Weigel file...')

    df2 = pd.read_csv( target + 'data.csv')
    
    n = 100
    x2 = np.zeros(n)
    y2 = np.zeros(n)
    z2 = np.zeros(n)
    
    x1 = np.zeros(n)
    y1 = np.zeros(n)
    z1 = np.zeros(n)
    
    jx2 = np.zeros(n)
    jy2 = np.zeros(n)
    jz2 = np.zeros(n)
    jMag2 = np.zeros(n)
    jphi2 = np.zeros(n)
    
    dBx2 = np.zeros(n)
    dBy2 = np.zeros(n)
    dBz2 = np.zeros(n)
    dBMag2 = np.zeros(n)

    ux2 = np.zeros(n)
    uy2 = np.zeros(n)
    uz2 = np.zeros(n)
    uMag2 = np.zeros(n)

    bx2 = np.zeros(n)
    by2 = np.zeros(n)
    bz2 = np.zeros(n)

    jx1 = np.zeros(n)
    jy1 = np.zeros(n)
    jz1 = np.zeros(n)
    jMag1 = np.zeros(n)
    jphi1 = np.zeros(n)
    
    dBx1 = np.zeros(n)
    dBy1 = np.zeros(n)
    dBz1 = np.zeros(n)
    dBMag1 = np.zeros(n)

    ux1 = np.zeros(n)
    uy1 = np.zeros(n)
    uz1 = np.zeros(n)
    uMag1 = np.zeros(n)

    bx1 = np.zeros(n)
    by1 = np.zeros(n)
    bz1 = np.zeros(n)

    i = 0
    reltol = 0.0001
    
    while i < n:
        df2_sample = df2.sample()
        
        x2[i] = df2_sample['CellCenter_0'].iloc[-1]
        y2[i] = df2_sample['CellCenter_1'].iloc[-1]
        z2[i] = df2_sample['CellCenter_2'].iloc[-1]
                
        df1a = df1[ np.isclose( df1['x'], x2[i], rtol = reltol ) ]
        logging.info(f'Returned x cut...{len(df1a)}')

        if( len(df1a) < 1):
            logging.info(f'Returned bad df1a...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
    
        df1b = df1a[ np.isclose( df1a['y'], y2[i], rtol = reltol ) ]
        logging.info(f'Returned y cut...{len(df1b)}')

        if( len(df1b) < 1):
            logging.info(f'Returned bad df1b...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
    
        df1c = df1b[ np.isclose( df1b['z'], z2[i], rtol = reltol ) ]
        logging.info(f'Returned z cut...{len(df1c)}')

        if( len(df1c) != 1):
            logging.info(f'Returned bad df1c...{i}, {x2[i]}, {y2[i]}, {z2[i]}')
            continue
        
        if( not np.isclose( df2_sample['j_0'].iloc[-1], df1c['jx'].iloc[-1] ) and
           not np.isclose( df2_sample['j_1'].iloc[-1], df1c['jy'].iloc[-1] ) and
           not np.isclose( df2_sample['j_2'].iloc[-1], df1c['jz'].iloc[-1] ) ):
            logging.info(f'Current densities are different: {i}')
            continue
            
        jx2[i] = df2_sample['j_0'].iloc[-1]
        jy2[i] = df2_sample['j_1'].iloc[-1]
        jz2[i] = df2_sample['j_2'].iloc[-1]
        jMag2[i] = df2_sample['j_Magnitude'].iloc[-1]
        jphi2[i] = df2_sample['j_phi'].iloc[-1]
        
        dBx2[i] = df2_sample['dB_0'].iloc[-1]
        dBy2[i] = df2_sample['dB_1'].iloc[-1]
        dBz2[i] = df2_sample['dB_2'].iloc[-1]
        dBMag2[i] = df2_sample['dB_Magnitude'].iloc[-1]

        ux2[i] = df2_sample['u_0'].iloc[-1]
        uy2[i] = df2_sample['u_1'].iloc[-1]
        uz2[i] = df2_sample['u_2'].iloc[-1]
        uMag2[i] = df2_sample['u_Magnitude'].iloc[-1]

        bx2[i] = df2_sample['b_0'].iloc[-1]
        by2[i] = df2_sample['b_1'].iloc[-1]
        bz2[i] = df2_sample['b_2'].iloc[-1]

        jx1[i] = df1c['jx'].iloc[-1]
        jy1[i] = df1c['jy'].iloc[-1]
        jz1[i] = df1c['jz'].iloc[-1]
        jMag1[i] = df1c['jMag'].iloc[-1]
        jphi1[i] = df1c['jphi'].iloc[-1]
        
        dBx1[i] = df1c['dBx'].iloc[-1]
        dBy1[i] = df1c['dBy'].iloc[-1]
        dBz1[i] = df1c['dBz'].iloc[-1]
        dBMag1[i] = df1c['dBmag'].iloc[-1]
        
        ux1[i] = df1c['ux'].iloc[-1]
        uy1[i] = df1c['uy'].iloc[-1]
        uz1[i] = df1c['uz'].iloc[-1]
        uMag1[i] = df1c['uMag'].iloc[-1]

        bx1[i] = df1c['bx'].iloc[-1]
        by1[i] = df1c['by'].iloc[-1]
        bz1[i] = df1c['bz'].iloc[-1]

        x1[i] = df1c['x'].iloc[-1]
        y1[i] = df1c['y'].iloc[-1]
        z1[i] = df1c['z'].iloc[-1]

        i = i + 1
     
    plt.subplot(1,3,1).scatter(x=x1, y=x2, s=1)
    plt.xlabel(r'$x$ (Dean)')
    plt.ylabel(r'$x$ (Bob)')
 
    plt.subplot(1,3,2).scatter(x=y1, y=y2, s=1)
    plt.xlabel(r'$y$ (Dean)')
    plt.ylabel(r'$y$ (Bob)')
 
    plt.subplot(1,3,3).scatter(x=z1, y=z2, s=1)
    plt.xlabel(r'$z$ (Dean)')
    plt.ylabel(r'$z$ (Bob)')

    plt.figure()     
 
    plt.subplot(1,3,1).scatter(x=dBx1, y=dBx2, s=1)
    plt.xlabel(r'$\delta B_x$ (Dean)')
    plt.ylabel(r'$\delta B_x$ (Bob)')
 
    plt.subplot(1,3,2).scatter(x=dBy1, y=dBy2, s=1)
    plt.xlabel(r'$\delta B_y$ (Dean)')
    plt.ylabel(r'$\delta B_y$ (Bob)')
 
    plt.subplot(1,3,3).scatter(x=dBz1, y=dBz2, s=1)
    plt.xlabel(r'$\delta B_z$ (Dean)')
    plt.ylabel(r'$\delta B_z$ (Bob)')
  
    plt.figure()

    plt.subplot(1,3,1).scatter(x=jx1, y=jx2, s=1)
    plt.xlabel(r'$j_x$ (Dean)')
    plt.ylabel(r'$j_x$ (Bob)')
    
    plt.subplot(1,3,2).scatter(x=jy1, y=jy2, s=1)
    plt.xlabel(r'$j_y$ (Dean)')
    plt.ylabel(r'$j_y$ (Bob)')
    
    plt.subplot(1,3,3).scatter(x=jz1, y=jz2, s=1)
    plt.xlabel(r'$j_z$ (Dean)')
    plt.ylabel(r'$j_z$ (Bob)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=ux1, y=ux2, s=1)
    plt.xlabel(r'$u_x$ (Dean)')
    plt.ylabel(r'$u_x$ (Bob)')
    
    plt.subplot(1,3,2).scatter(x=uy1, y=uy2, s=1)
    plt.xlabel(r'$u_y$ (Dean)')
    plt.ylabel(r'$u_y$ (Bob)')
    
    plt.subplot(1,3,3).scatter(x=uz1, y=uz2, s=1)
    plt.xlabel(r'$u_z$ (Dean)')
    plt.ylabel(r'$u_z$ (Bob)')
    
    plt.figure()

    plt.subplot(1,3,1).scatter(x=bx1, y=bx2, s=1)
    plt.xlabel(r'$b_x$ (Dean)')
    plt.ylabel(r'$b_x$ (Bob)')
    
    plt.subplot(1,3,2).scatter(x=by1, y=by2, s=1)
    plt.xlabel(r'$b_y$ (Dean)')
    plt.ylabel(r'$b_y$ (Bob)')
    
    plt.subplot(1,3,3).scatter(x=bz1, y=bz2, s=1)
    plt.xlabel(r'$b_z$ (Dean)')
    plt.ylabel(r'$b_z$ (Bob)')
    
    plt.figure()
    
    plt.subplot(1,3,1).scatter(x=jphi1, y=jphi2, s=1)
    plt.xlabel(r'$| j_\phi |$ (Dean)')
    plt.ylabel(r'$| j_\phi |$ (Bob)')
    

def process_data_with_cuts( base, dirpath = origin ):
    """Process data in BATSRUS file to create dataframe with calculated quantities,
    but in this case we perform some cuts on the data to isolate high current
    regions.  This cut data is used to determine the fraction of the total B
    field due to each set of currents
        
    Inputs:
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        b_all = Bz field based on summing all dB
        b_asym_jr = Bz field based on summing dB associated with the asymmetric
            jr region
        b_y_jphi = Bz field based on summing dB associated with the jphi vs y blob
        b_z_jphi = Bz field based on summing dB associated with the jphi vs z blob
        b_rest = Bz field based on summing dB associated with everything else,
            b_rest + b_z_jphi + b_y_jphi + b_aym_jr = b_all
    """
           
    logging.info('Parsing BATSRUS file...')
    
    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')
    
    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert( batsrus != None )
       
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    x = batsrus.data_arr[:,var_dict['x']][:]
    y = batsrus.data_arr[:,var_dict['y']][:]
    z = batsrus.data_arr[:,var_dict['z']][:]
        
    jx = batsrus.data_arr[:,var_dict['jx']][:]
    jy = batsrus.data_arr[:,var_dict['jy']][:]
    jz = batsrus.data_arr[:,var_dict['jz']][:]
        
    measure = batsrus.data_arr[:,var_dict['measure']][:]
    
    logging.info('Calculating delta B...')
    
    # Calculate new quantities
    r = ((X-x)**2+(Y-y)**2+(Z-z)**2)**(1/2)
    factor = 637.1*measure/r**3
    
    dBx = factor*( jy*(Z-z) - jz*(Y-y) )
    dBy = factor*( jz*(X-x) - jx*(Z-z) )
    dBz = factor*( jx*(Y-y) - jy*(X-x) )

    dBx[r < rCurrents] = 0.0
    dBy[r < rCurrents] = 0.0
    dBz[r < rCurrents] = 0.0

    jMag = np.sqrt( jx**2 + jy**2 + jz**2 )
 
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1]
    
    logging.info('Creating jr, jtheta, jphi dataframe...')
    
    # Make a new dataframe that we use for spherical coordinate transform
    # and separating the data into dayside and nightside regions
    
    df2 = pd.DataFrame()
    df2['x'] = deepcopy(x)
    df2['y'] = deepcopy(y)
    df2['z'] = deepcopy(z)
    df2['jMag'] = deepcopy(jMag)
    df2['jx'] = deepcopy(jx)
    df2['jy'] = deepcopy(jy)
    df2['jz'] = deepcopy(jz)
    df2['dBx'] = deepcopy(dBx)
    df2['dBy'] = deepcopy(dBy)
    df2['dBz'] = deepcopy(dBz)
    df2['r'] = deepcopy(r)
    
    # Keep only data outside of rCurrents
    df2 = df2[df2['r'] > rCurrents]

    # Transform the currents, j, into spherical coordinates
    
    # Determine theta and phi of the radius vector from the origin to the 
    # center of the cell
    df2['theta'] = np.arccos( df2['z']/df2['r'] )
    df2['phi'] = np.arctan2( df2['y'], df2['x'] )
    
    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df2['jr'] = df2['jx'] * np.sin(df2['theta']) * np.cos(df2['phi']) + \
        df2['jy'] * np.sin(df2['theta']) * np.sin(df2['phi']) + \
        df2['jz'] * np.cos(df2['theta'])
        
    df2['jtheta'] = df2['jx'] * np.cos(df2['theta']) * np.cos(df2['phi']) + \
        df2['jy'] * np.cos(df2['theta']) * np.sin(df2['phi']) - \
        df2['jz'] * np.sin(df2['theta'])
        
    df2['jphi'] = - df2['jx'] * np.sin(df2['phi']) + df2['jy'] * np.cos(df2['phi'])
    
    logging.info('Creating dataframes with extracted cuts...')
   
    #################################
    #################################    
    # Cut asymmetric jr vs y lobes
    df3 = df2.drop( df2[np.logical_and(df2['jr'].abs() > 0.007, df2['y'].abs() < 4)].index)
    #################################
    #################################
    # Cut jphi vs y blob
    df4 = df2.drop( df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
    #################################
    #################################
    # Cut jphi vs z blob
    df5 = df2.drop( df2[np.logical_and(df2['jphi'].abs() > 0.007, df2['z'].abs() < 2)].index)
    #################################
    #################################
    # Cut everything else
    df6 = df2.drop( df2[np.logical_and(df2['jphi'].abs() > 0.007, df2['z'].abs() < 2)].index)
    df6 = df6.drop( df6[np.logical_and(df6['jphi'] > 0.007, df6['jphi'] < 0.03)].index)
    df6 = df6.drop( df6[np.logical_and(df6['jr'].abs() > 0.007, df6['y'].abs() < 4)].index)
    #################################
    #################################
    
    logging.info('Calculate cumulative sums for dataframes for extracted cuts...')

    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.  
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.
    df2 = df2.sort_values(by='r', ascending=True)
    df2['dBxSum'] = df2['dBx'].cumsum()
    df2['dBySum'] = df2['dBy'].cumsum()
    df2['dBzSum'] = df2['dBz'].cumsum()
    df2['dBxSumMag'] = df2['dBxSum'].abs()
    df2['dBySumMag'] = df2['dBySum'].abs()
    df2['dBzSumMag'] = df2['dBzSum'].abs()
    df2['dBSumMag'] = (df2['dBxSum']**2 + df2['dBySum']**2 + df2['dBzSum']**2)**(1/2)

    df3 = df3.sort_values(by='r', ascending=True)
    df3['dBxSum'] = df3['dBx'].cumsum()
    df3['dBySum'] = df3['dBy'].cumsum()
    df3['dBzSum'] = df3['dBz'].cumsum()
    df3['dBxSumMag'] = df3['dBxSum'].abs()
    df3['dBySumMag'] = df3['dBySum'].abs()
    df3['dBzSumMag'] = df3['dBzSum'].abs()
    df3['dBSumMag'] = (df3['dBxSum']**2 + df3['dBySum']**2 + df3['dBzSum']**2)**(1/2)

    df4 = df4.sort_values(by='r', ascending=True)
    df4['dBxSum'] = df4['dBx'].cumsum()
    df4['dBySum'] = df4['dBy'].cumsum()
    df4['dBzSum'] = df4['dBz'].cumsum()
    df4['dBxSumMag'] = df4['dBxSum'].abs()
    df4['dBySumMag'] = df4['dBySum'].abs()
    df4['dBzSumMag'] = df4['dBzSum'].abs()
    df4['dBSumMag'] = (df4['dBxSum']**2 + df4['dBySum']**2 + df4['dBzSum']**2)**(1/2)

    df5 = df5.sort_values(by='r', ascending=True)
    df5['dBxSum'] = df5['dBx'].cumsum()
    df5['dBySum'] = df5['dBy'].cumsum()
    df5['dBzSum'] = df5['dBz'].cumsum()
    df5['dBxSumMag'] = df5['dBxSum'].abs()
    df5['dBySumMag'] = df5['dBySum'].abs()
    df5['dBzSumMag'] = df5['dBzSum'].abs()
    df5['dBSumMag'] = (df5['dBxSum']**2 + df5['dBySum']**2 + df5['dBzSum']**2)**(1/2)

    df6 = df6.sort_values(by='r', ascending=True)
    df6['dBxSum'] = df6['dBx'].cumsum()
    df6['dBySum'] = df6['dBy'].cumsum()
    df6['dBzSum'] = df6['dBz'].cumsum()
    df6['dBxSumMag'] = df6['dBxSum'].abs()
    df6['dBySumMag'] = df6['dBySum'].abs()
    df6['dBzSumMag'] = df6['dBzSum'].abs()
    df6['dBSumMag'] = (df6['dBxSum']**2 + df6['dBySum']**2 + df6['dBzSum']**2)**(1/2)

    return df2['dBzSum'].iloc[-1], \
            df2['dBzSum'].iloc[-1] - df3['dBzSum'].iloc[-1], \
            df2['dBzSum'].iloc[-1] - df4['dBzSum'].iloc[-1], \
            df2['dBzSum'].iloc[-1] - df5['dBzSum'].iloc[-1], \
            df6['dBzSum'].iloc[-1]
   
def loop_thru_cuts( files ):
    
    from datetime import time 
    plt.rcParams["figure.figsize"] = [3.6,3.2]

    n = len(files)
    
    b_all = [None] * n
    b_asym_jr = [None] * n
    b_y_jphi = [None] * n
    b_z_jphi = [None] * n
    b_rest = [None] * n
    b_times = [None] * n
    b_index = [None] * n
    b_test = [None] * n
    
    for i in range(n):
        # Create the title that we'll use in the graphics
        words = files[i].split('-')
        t = int(words[1])
        h = t//10000
        m = (t%10000) // 100
        logging.info(f'Time: {t} Hours: {h} Minutes: {m}')
        b_times[i] = time(h, m)
        b_index[i] = i

        b_all[i], b_asym_jr[i], b_y_jphi[i], b_z_jphi[i], b_rest[i] = \
            process_data_with_cuts(base = files[i])
    
        b_test[i] = b_all[i] - b_rest[i] - b_asym_jr[i] - b_y_jphi[i] - b_z_jphi[i]
    
    # from matplotlib.dates import date2num
    # b_plt_times = date2num( np.array(b_times) )
    
    plt.plot(b_index, b_all,     ls='solid', color='blue')
    plt.plot(b_index, b_asym_jr, ls='dashed', color='red')
    plt.plot(b_index, b_y_jphi,  ls='dashdot', color='green')
    plt.plot(b_index, b_z_jphi,  ls='dotted', color='black')
    plt.plot(b_index, b_rest,    ls='solid', color='yellow')
    plt.plot(b_index, b_test,    ls='solid', color='red')
    plt.xlabel(r'index')
    plt.ylabel(r'$B_z$')
    plt.ylim(-dB_sum_limits[1],dB_sum_limits[1])
    plt.legend(['All', r'Asym $j_r$', r'y $j_\phi$', r'z $j_\phi$', 'Remainder', 'Test (zero)'])
    
    return


def get_files( basedir = origin, base = '3d__*'):
    import os
    import glob
    
    # Create a list of files that we will process
    # Look in the basedir directory.  Get list of file basenames
    
    # In this version, we find all of the base + '.out' files
    # and retrieve their basenames
    os.chdir( basedir )
    
    l = glob.glob( base + '.out')
    
    # Strip off extension
    for i in range(len(l)):
        l[i] = (l[i].split('.'))[0]
    
    l.sort()
    
    return l

def get_files_unconverted( directory = 'png-dBmagNorm-uMag-night', 
                          orgdir = origin, tgtdir = target, base = '3d__*' ):
    import os
    import glob
    
    # Create a list of files that we will process
    # Compare the directory and the origin.  Get list of file basenames
    
    # In this version we compare the list of out files and png files
    # to determine what has already been processed.  Look for all *.out
    # files and remove from the list (l1) all of them that have
    # already been converted to .png files
    os.chdir( orgdir )
    l1 = glob.glob( base + '.out')
    
    # Look at the png files in directory
    os.chdir( tgtdir + directory )
    l2 = glob.glob( base + '.png' )

    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    for i in range(len(l2)):
        l2[i] = (l2[i].split('.'))[0]

    for i in l2:
        l1.remove(i)
    
    l1.sort()
        
    return l1

if __name__ == "__main__":
    # files = get_files()
    # files = get_files_unconverted( directory = 'png-combined-sum-dB-cuts' )
    
    # logging.info('Num. of files: ' + str(len(files)))
 
    # for i in range(len(files)):
        # process_data(base = files[i])
        # process_data_with_cuts(base = files[i])
        # process_data_dbl_chk(base = files[i])
    
    process_data_compare_weigel()
    # loop_thru_cuts(files)

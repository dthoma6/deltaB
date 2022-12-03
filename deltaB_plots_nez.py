#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: dean
"""
################################################################################
################################################################################
# Derived from deltaB_plots.py. Only signficant difference is rather than
# plotting B in x,y,z components, its in north-east-zenith coordinates
################################################################################
################################################################################

# Info on divB_simple1 runs available at:
#
# https://ccmc.gsfc.nasa.gov/results/viewrun.php?domain=GM&runnumber=Brian_Curtis_042213_7
#
# https://ccmc.gsfc.nasa.gov/RoR_WWW/GM/SWMF/2013/Brian_Curtis_042213_7/Brian_Curtis_042213_7_sw.gif
#

# origin and target define where input data and output plots are stored
import logging
from copy import deepcopy
import swmfio
from os import makedirs
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
origin = '/Volumes/Physics HD v2/divB_simple1/GM/'
target = '/Volumes/Physics HD v2/divB_simple1/plots/'

# rCurrents define range from earth center below which results are not valid
# measured in Re units
rCurrents = 3

# Initialize useful variables
#(X,Y,Z) = (1.0, 0.0, 0.0)

# range of values seen in each variable, used to plot graphs
rho_limits = [10**-2, 10**2]
p_limits = [10**-5, 10**2]
jMag_limits = [10**-11, 10**0]
j_limits = [-0.3, 0.3]
uMag_limits = [10**-3, 10**4]
u_limits = [-1100, 1100]
dBNorm_limits = [10**-15, 10**-1]

dBx_sum_limits = [-0.4, 0.4]
dBy_sum_limits = [-0.4, 0.4]
dBz_sum_limits = [-50, 50]
dBp_sum_limits = [-50, 50]
dB_sum_limits = [0, 50]

plot3d_limits = [-10, 10]


# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.8, 7.2]
# plt.rcParams["figure.figsize"] = [3.6,3.2]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 7
plt.rcParams['axes.grid'] = True


def nez(time, pos, csys):
    """Unit vectors in geographic north, east, and zenith directions
    from Weigel"""

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


def create_directory(folder):
    """ If directory for output files does not exist, create it

    Inputs:
        folder = basename of folder.  Complete path to folder is:
            target + folder
    Outputs:
        None 
     """

    logging.info('Looking for directory: ' + target + folder)
    if not exists(target + folder):
        logging.info('Creating directory: ' + target + folder)
        makedirs(target + folder)
    return


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


def plot_db_Norm_r(df, title, base):
    """Plot various forms of the magnitude of dB in each cell versus radius r.
    In this procedure, the dB values are normalized by cell volume

    Inputs:
        df = dataframe with BATSRUS data and calculated variables
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    # Plot dB (normalized by cell volume) as a function of range r
    plt.subplot(2, 4, 1).scatter(x=df['r'], y=df['dBxNorm'], s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_x |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15, 10**-1)

    plt.subplot(2, 4, 2).scatter(x=df['r'], y=df['dByNorm'], s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_y |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15, 10**-1)

    plt.subplot(2, 4, 3).scatter(x=df['r'], y=df['dBzNorm'], s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B_z |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15, 10**-1)

    plt.subplot(2, 4, 4).scatter(x=df['r'], y=df['dBmagNorm'], s=1)
    plt.yscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$| \delta B |$ (Norm Cell Vol)')
    plt.title(title)
    plt.ylim(10**-15, 10**-1)

    fig = plt.gcf()
    create_directory('png-combined-dBNorm-r/')
    logging.info(f'Saving {base} combined dBNorm plot')
    fig.savefig(target + 'png-combined-dBNorm-r/' +
                base + '.out.combined-dBNorm-r.png')
    plt.close(fig)

    return


def plot_dBnorm_day_night(df_day, df_night, title, base):
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
    df_day.plot.scatter(x='rho', y='dBmagNorm',
                        ax=plt.subplot(2, 4, 1),
                        logx=True,
                        logy=True,
                        xlim=rho_limits,
                        ylim=dBNorm_limits,
                        xlabel=r'$\rho$',
                        ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='rho', y='dBmagNorm',
                          ax=plt.subplot(2, 4, 5),
                          logx=True,
                          logy=True,
                          xlim=rho_limits,
                          ylim=dBNorm_limits,
                          xlabel=r'$\rho$',
                          ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                          title='Night ' + title,
                          s=1)

    # Plot dBmagNorm as a function of p
    df_day.plot.scatter(x='p', y='dBmagNorm',
                                 ax=plt.subplot(2, 4, 2),
                                 logx=True,
                                 logy=True,
                                 xlim=p_limits,
                                 ylim=dBNorm_limits,
                                 xlabel=r'$p$',
                                 ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                                 title='Day ' + title,
                                 s=1)

    df_night.plot.scatter(x='p', y='dBmagNorm',
                          ax=plt.subplot(2, 4, 6),
                          logx=True,
                          logy=True,
                          xlim=p_limits,
                          ylim=dBNorm_limits,
                          xlabel=r'$p$',
                          ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                          title='Night ' + title,
                          s=1)

    # Plot dBmagNorm as a function of jMag
    df_day.plot.scatter(x='jMag', y='dBmagNorm',
                        ax=plt.subplot(2, 4, 3),
                        logx=True,
                        logy=True,
                        xlim=jMag_limits,
                        ylim=dBNorm_limits,
                        xlabel=r'$| j |$',
                        ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='jMag', y='dBmagNorm',
                          ax=plt.subplot(2, 4, 7),
                          logx=True,
                          logy=True,
                          xlim=jMag_limits,
                          ylim=dBNorm_limits,
                          xlabel=r'$| j |$',
                          ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                          title='Night ' + title,
                          s=1)

    # Plot dBmagNorm as a function of uMag
    df_day.plot.scatter(x='uMag', y='dBmagNorm',
                        ax=plt.subplot(2, 4, 4),
                        logx=True,
                        logy=True,
                        xlim=uMag_limits,
                        ylim=dBNorm_limits,
                        xlabel=r'$| u |$',
                        ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='uMag', y='dBmagNorm',
                          ax=plt.subplot(2, 4, 8),
                          logx=True,
                          logy=True,
                          xlim=uMag_limits,
                          ylim=dBNorm_limits,
                          xlabel=r'$| u |$',
                          ylabel=r'$| \delta B |$ (Norm Cell Vol)',
                          title='Night ' + title,
                          s=1)

    fig = plt.gcf()
    create_directory('png-combined-dBNorm-day-night/')
    logging.info(f'Saving {base} combined dBNorm day-night plot')
    fig.savefig(target + 'png-combined-dBNorm-day-night/' +
                base + '.out.png-combined-dBNorm-day-night.png')
    plt.close(fig)

    return


def plot_cumulative_B(df_r, title, base):
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

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBxSum',
                      ax=plt.subplot(2, 4, 1),
                      xlim=[1, 1000],
                      ylim=dBx_sum_limits,
                      logx=True,
                      title=title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_x $')

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBySum',
                      ax=plt.subplot(2, 4, 2),
                      xlim=[1, 1000],
                      ylim=dBy_sum_limits,
                      logx=True,
                      title=title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_y$')

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBzSum',
                      ax=plt.subplot(2, 4, 3),
                      xlim=[1, 1000],
                      ylim=dBz_sum_limits,
                      logx=True,
                      title=title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_z$')

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBSumMag',
                      ax=plt.subplot(2, 4, 4),
                      xlim=[1, 1000],
                      ylim=dB_sum_limits,
                      logx=True,
                      title=title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$| \Sigma_r \delta B |$')

    fig = plt.gcf()
    create_directory('png-combined-sum-dB-r/')
    logging.info(f'Saving {base} combined dB plot')
    fig.savefig(target + 'png-combined-sum-dB-r/' +
                base + '.out.combined-sum-dB-r.png')
    plt.close(fig)

    return


def plot_cumulative_B_para_anti(df_r_para, df_r_anti, title, base):
    """Plot various forms of the cumulative sum of dB in each cell versus 
        range r.  These plots examine the contributions of the currents
        parallel to the magnetic field and anti-parallel.  
        To generate the cumulative sum, we order the cells in terms of
        range r from the earth's center.  We start with a small sphere and 
        vector sum all of the dB contributions inside the sphere.  Expand the
        sphere slightly and resum.  Repeat until all cells are in the sum.

    Inputs:
        df_r_para, df_r_anti = dataframe containing cumulative sums ordered from small r to 
            large r.  One for currents parallel to the B field, and one anti-parallel.
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    # Plot the cummulative sum of dB as a function of r
    df_r_para.plot.scatter(x='r', y='dBxSum',
                           ax=plt.subplot(2, 4, 1),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_x (j_\parallel)$')

    df_r_anti.plot.scatter(x='r', y='dBxSum',
                           ax=plt.subplot(2, 4, 5),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'Anti-$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_x (j_{anti-\parallel})$')

    # Plot the cummulative sum of dB as a function of r
    df_r_para.plot.scatter(x='r', y='dBySum',
                           ax=plt.subplot(2, 4, 2),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_y (j_\parallel)$')

    df_r_anti.plot.scatter(x='r', y='dBySum',
                           ax=plt.subplot(2, 4, 6),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'Anti-$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_y (j_{anti-\parallel})$')

    # Plot the cummulative sum of dB as a function of r
    df_r_para.plot.scatter(x='r', y='dBzSum',
                           ax=plt.subplot(2, 4, 3),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_z (j_\parallel)$')

    df_r_anti.plot.scatter(x='r', y='dBzSum',
                           ax=plt.subplot(2, 4, 7),
                           xlim=[1, 1000],
                           ylim=dBp_sum_limits,
                           logx=True,
                           title=r'Anti-$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$\Sigma_r \delta B_z (j_{anti-\parallel})$')

    # Plot the cummulative sum of dB as a function of r
    df_r_para.plot.scatter(x='r', y='dBSumMag',
                           ax=plt.subplot(2, 4, 4),
                           xlim=[1, 1000],
                           ylim=dB_sum_limits,
                           logx=True,
                           title=r'$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$| \Sigma_r \delta B (j_\parallel)|$')

    df_r_anti.plot.scatter(x='r', y='dBSumMag',
                           ax=plt.subplot(2, 4, 8),
                           xlim=[1, 1000],
                           ylim=dB_sum_limits,
                           logx=True,
                           title=r'Anti-$\parallel$ ' + title,
                           s=1,
                           xlabel='$r/R_E$',
                           ylabel=r'$| \Sigma_r \delta B (j_{anti-\parallel})|$')

    fig = plt.gcf()
    create_directory('png-combined-sum-dB-para-anti-r/')
    logging.info(f'Saving {base} combined dB parallel/anti-parallel plot')
    fig.savefig(target + 'png-combined-sum-dB-para-anti-r/' +
                base + '.out.combined-sum-dB-para-anti-r.png')
    plt.close(fig)

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

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBparallelxSum',
                      ax=plt.subplot(2, 4, 1),
                      xlim=[1, 1000],
                      ylim=dBx_sum_limits,
                      logx=True,
                      title=r'$\parallel$ ' + title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_x (j_\parallel)$')

    # df_r.plot.scatter(x='r', y='dBperpendicularxSum',
    #                   ax=plt.subplot(2, 4, 5),
    #                   xlim=[1, 1000],
    #                   ylim=dBx_sum_limits,
    #                   logx=True,
    #                   title=r'$\perp$ ' + title,
    #                   s=1,
    #                   xlabel='$r/R_E$',
    #                   ylabel=r'$\Sigma_r \delta B_x (j_{\perp})$')
    
    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularxSum'], s=1, color='blue')
    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularphixSum'], s=1, color='red')
    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularphiresxSum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_x (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBx_sum_limits[0], dBx_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residue}$'])

    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBparallelySum',
                      ax=plt.subplot(2, 4, 2),
                      xlim=[1, 1000],
                      ylim=dBy_sum_limits,
                      logx=True,
                      title=r'$\parallel$ ' + title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_y (j_\parallel)$')

    # df_r.plot.scatter(x='r', y='dBperpendicularySum',
    #                   ax=plt.subplot(2, 4, 6),
    #                   xlim=[1, 1000],
    #                   ylim=dBy_sum_limits,
    #                   logx=True,
    #                   title=r'$\perp$ ' + title,
    #                   s=1,
    #                   xlabel='$r/R_E$',
    #                   ylabel=r'$\Sigma_r \delta B_y (j_{\perp})$')

    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularySum'], s=1, color='blue')
    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularphiySum'], s=1, color='red')
    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularphiresySum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_y (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBy_sum_limits[0], dBy_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residue}$'])


    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBparallelzSum',
                      ax=plt.subplot(2, 4, 3),
                      xlim=[1, 1000],
                      ylim=dBz_sum_limits,
                      logx=True,
                      title=r'$\parallel$ ' + title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$\Sigma_r \delta B_z (j_\parallel)$')

    # df_r.plot.scatter(x='r', y='dBperpendicularzSum',
    #                   ax=plt.subplot(2, 4, 7),
    #                   xlim=[1, 1000],
    #                   ylim=dBz_sum_limits,
    #                   logx=True,
    #                   title=r'$\perp$ ' + title,
    #                   s=1,
    #                   xlabel='$r/R_E$',
    #                   ylabel=r'$\Sigma_r \delta B_z (j_{\perp})$')

    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularzSum'], s=1, color='blue')
    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularphizSum'], s=1, color='red')
    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularphireszSum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_z (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBz_sum_limits[0], dBz_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residue}$'])


    # Plot the cummulative sum of dB as a function of r
    df_r.plot.scatter(x='r', y='dBparallelSumMag',
                      ax=plt.subplot(2, 4, 4),
                      xlim=[1, 1000],
                      ylim=dB_sum_limits,
                      logx=True,
                      title=r'$\parallel$ ' + title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$| \Sigma_r \delta B (j_\parallel)|$')

    df_r.plot.scatter(x='r', y='dBperpendicularSumMag',
                      ax=plt.subplot(2, 4, 8),
                      xlim=[1, 1000],
                      ylim=dB_sum_limits,
                      logx=True,
                      title=r'$\perp$ ' + title,
                      s=1,
                      xlabel='$r/R_E$',
                      ylabel=r'$| \Sigma_r \delta B (j_{\perp})|$')

    fig = plt.gcf()
    create_directory('png-combined-sum-dB-para-perp-comp-r/')
    logging.info(f'Saving {base} combined dB parallel/perpendicular plot')
    fig.savefig(target + 'png-combined-sum-dB-para-perp-comp-r/' +
                base + '.out.combined-sum-dB-para-perp-comp-r.png')
    plt.close(fig)

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

    # Plot rho as a function of range r
    df_day.plot.scatter(x='r', y='rho',
                        ax=plt.subplot(2, 4, 1),
                        logx=True,
                        logy=True,
                        xlim=[1, 10**3],
                        ylim=rho_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$\rho$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='rho',
                          ax=plt.subplot(2, 4, 5),
                          logx=True,
                          logy=True,
                          xlim=[1, 10**3],
                          ylim=rho_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$\rho$',
                          title='Night ' + title,
                          s=1)

    # Plot p as a function of range r
    df_day.plot.scatter(x='r', y='p',
                        ax=plt.subplot(2, 4, 2),
                        logx=True,
                        logy=True,
                        xlim=[1, 10**3],
                        ylim=p_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$p$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='p',
                          ax=plt.subplot(2, 4, 6),
                          logx=True,
                          logy=True,
                          xlim=[1, 10**3],
                          ylim=p_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$p$',
                          title='Night ' + title,
                          s=1)

    # Plot jMag as a function of range r
    df_day.plot.scatter(x='r', y='jMag',
                        ax=plt.subplot(2, 4, 3),
                        logx=True,
                        logy=True,
                        xlim=[1, 10**3],
                        ylim=jMag_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$|j|$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='jMag',
                          ax=plt.subplot(2, 4, 7),
                          logx=True,
                          logy=True,
                          xlim=[1, 10**3],
                          ylim=jMag_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$|j|$',
                          title='Night ' + title,
                          s=1)

    # Plot uMag as a function of range r
    df_day.plot.scatter(x='r', y='uMag',
                        ax=plt.subplot(2, 4, 4),
                        logx=True,
                        logy=True,
                        xlim=[1, 10**3],
                        ylim=uMag_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$|u|$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='uMag',
                          ax=plt.subplot(2, 4, 8),
                          logx=True,
                          logy=True,
                          xlim=[1, 10**3],
                          ylim=uMag_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$|u|$',
                          title='Night ' + title,
                          s=1)

    fig = plt.gcf()
    create_directory('png-combined-day-night/')
    logging.info(f'Saving {base} combined day-night plot')
    fig.savefig(target + 'png-combined-day-night/' +
                base + '.out.png-combined-day-night.png')
    plt.close(fig)

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

    # Plot jx as a function of range r
    df_day.plot.scatter(x='r', y='jx',
                        ax=plt.subplot(2, 4, 1),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=j_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$j_x$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='jx',
                          ax=plt.subplot(2, 4, 5),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=j_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$j_x$',
                          title='Night ' + title,
                          s=1)

    # Plot jy as a function of range r
    df_day.plot.scatter(x='r', y='jy',
                        ax=plt.subplot(2, 4, 2),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=j_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$j_y$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='jy',
                          ax=plt.subplot(2, 4, 6),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=j_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$j_y$',
                          title='Night ' + title,
                          s=1)

    # Plot jz as a function of range r
    df_day.plot.scatter(x='r', y='jz',
                        ax=plt.subplot(2, 4, 3),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=j_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$j_z$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='jz',
                          ax=plt.subplot(2, 4, 7),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=j_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$j_z$',
                          title='Night ' + title,
                          s=1)

    # Plot jMag as a function of range r
    df_day.plot.scatter(x='r', y='jMag',
                        ax=plt.subplot(2, 4, 4),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=j_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$|j|$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='jMag',
                          ax=plt.subplot(2, 4, 8),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=j_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$|j|$',
                          title='Night ' + title,
                          s=1)

    fig = plt.gcf()
    create_directory('png-combined-jxyz-day-night/')
    logging.info(f'Saving {base} combined jx, jy, jz day-night plot')
    fig.savefig(target + 'png-combined-jxyz-day-night/' +
                base + '.out.png-combined-jxyz-day-night.png')
    plt.close(fig)

    return


def plot_jx_jy_jz_vs_x(df, title, base):
    """Plot jx, jy, jz  in each cell versus x.  

    Inputs:
        df = dataframe containing jx, jy, jz and x
        title = title for plots
        base = basename of file where plot will be stored
    Outputs:
        None 
     """

    # Plot jx as a function of range x
    df.plot.scatter(x='x', y='jx',
                    ax=plt.subplot(2, 4, 1),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_x$',
                    title=title,
                    s=1)

    df.plot.scatter(x='x', y='jx',
                    ax=plt.subplot(2, 4, 5),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_x$',
                    title=title,
                    s=1)

    # Plot jy as a function of range r
    df.plot.scatter(x='x', y='jy',
                    ax=plt.subplot(2, 4, 2),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_y$',
                    title=title,
                    s=1)

    df.plot.scatter(x='x', y='jy',
                    ax=plt.subplot(2, 4, 6),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_y$',
                    title=title,
                    s=1)

    # Plot jz as a function of range x
    df.plot.scatter(x='x', y='jz',
                    ax=plt.subplot(2, 4, 3),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_z$',
                    title=title,
                    s=1)

    df.plot.scatter(x='x', y='jz',
                    ax=plt.subplot(2, 4, 7),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$j_z$',
                    title=title,
                    s=1)

    # Plot jMag and r as a function of range x
    df.plot.scatter(x='x', y='jMag',
                    ax=plt.subplot(2, 4, 4),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$|j|$',
                    title=title,
                    s=1)

    df.plot.scatter(x='x', y='r',
                    ax=plt.subplot(2, 4, 8),
                    xlim=[-300, 300],
                    ylim=[-300, 300],
                    xlabel=r'$ x/R_E $',
                    ylabel=r'$r/R_E$',
                    title=title,
                    s=1)

    fig = plt.gcf()
    create_directory('png-combined-jxyz-x/')
    logging.info(f'Saving {base} combined jx, jy, jz vs x plot')
    fig.savefig(target + 'png-combined-jxyz-x/' +
                base + '.out.png-combined-jxyz-x.png')
    plt.close(fig)

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

    # Plot jr as a function of range x
    df.plot.scatter(x=coord, y='jr',
                    ax=plt.subplot(2, 4, 1),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_r$',
                    title=title,
                    s=1)

    df.plot.scatter(x=coord, y='jr',
                    ax=plt.subplot(2, 4, 5),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_r$',
                    title=title,
                    s=1)

    # Plot jtheta as a function of range r
    df.plot.scatter(x=coord, y='jtheta',
                    ax=plt.subplot(2, 4, 2),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_\theta$',
                    title=title,
                    s=1)

    df.plot.scatter(x=coord, y='jtheta',
                    ax=plt.subplot(2, 4, 6),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_\theta$',
                    title=title,
                    s=1)

    # Plot jphi as a function of range x
    df.plot.scatter(x=coord, y='jphi',
                    ax=plt.subplot(2, 4, 3),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_\phi$',
                    title=title,
                    s=1)

    df.plot.scatter(x=coord, y='jphi',
                    ax=plt.subplot(2, 4, 7),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_\phi$',
                    title=title,
                    s=1)

    # Plot jMag as a function of range x
    df.plot.scatter(x=coord, y='jMag',
                    ax=plt.subplot(2, 4, 4),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$|j|$',
                    title=title,
                    s=1)

    fig = plt.gcf()
    create_directory('png-combined-jrtp-'+cut+coord+'/')
    logging.info(
        f'Saving {base} combined jr, jtheta, jphi vs ' + coord + ' plot')
    fig.savefig(target + 'png-combined-jrtp-'+cut+coord+'/' +
                base + '.out.'+'png-combined-jrtp-'+cut+coord+'.png')
    plt.close(fig)

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

    # Plot jparallel as a function of range x
    df.plot.scatter(x=coord, y='jparallelMag',
                    ax=plt.subplot(2, 4, 1),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_{\parallel}$',
                    title=title,
                    s=1)

    df.plot.scatter(x=coord, y='jparallelMag',
                    ax=plt.subplot(2, 4, 5),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$j_{\parallel}$',
                    title=title,
                    s=1)

    # Plot jperpendicularMag as a function of range r
    df.plot.scatter(x=coord, y='jperpendicularMag',
                    ax=plt.subplot(2, 4, 2),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$| j_{\perp} |$',
                    title=title,
                    s=1)

    df.plot.scatter(x=coord, y='jperpendicularMag',
                    ax=plt.subplot(2, 4, 6),
                    xlim=[-20, 20],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$| j_{\perp} |$',
                    title=title,
                    s=1)

    # Plot jMag as a function of range x
    df.plot.scatter(x=coord, y='jMag',
                    ax=plt.subplot(2, 4, 4),
                    xlim=[-300, 300],
                    ylim=j_limits,
                    xlabel=r'$ ' + coord + '/R_E $',
                    ylabel=r'$|j|$',
                    title=title,
                    s=1)

    fig = plt.gcf()
    create_directory('png-combined-jpp-'+cut+coord+'/')
    logging.info(
        f'Saving {base} combined jparallel, jperpendicular vs ' + coord + ' plot')
    fig.savefig(target + 'png-combined-jpp-'+cut+coord+'/' +
                base + '.out.'+'png-combined-jpp-'+cut+coord+'.png')
    plt.close(fig)

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

    # Plot ux as a function of range r
    df_day.plot.scatter(x='r', y='ux',
                        ax=plt.subplot(2, 4, 1),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=u_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$u_x$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='ux',
                          ax=plt.subplot(2, 4, 5),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=u_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$u_x$',
                          title='Night ' + title,
                          s=1)

    # Plot uy as a function of range r
    df_day.plot.scatter(x='r', y='uy',
                        ax=plt.subplot(2, 4, 2),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=u_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$u_y$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='uy',
                          ax=plt.subplot(2, 4, 6),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=u_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$u_y$',
                          title='Night ' + title,
                          s=1)

    # Plot uz as a function of range r
    df_day.plot.scatter(x='r', y='uz',
                        ax=plt.subplot(2, 4, 3),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=u_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$u_z$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='uz',
                          ax=plt.subplot(2, 4, 7),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=u_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$u_z$',
                          title='Night ' + title,
                          s=1)

    # Plot uMag as a function of range r
    df_day.plot.scatter(x='r', y='uMag',
                        ax=plt.subplot(2, 4, 4),
                        logx=True,
                        xlim=[1, 10**3],
                        ylim=u_limits,
                        xlabel=r'$ r/R_E $',
                        ylabel=r'$|u|$',
                        title='Day ' + title,
                        s=1)

    df_night.plot.scatter(x='r', y='uMag',
                          ax=plt.subplot(2, 4, 8),
                          logx=True,
                          xlim=[1, 10**3],
                          ylim=u_limits,
                          xlabel=r'$ r/R_E $',
                          ylabel=r'$|u|$',
                          title='Night ' + title,
                          s=1)

    fig = plt.gcf()
    create_directory('png-combined-uxyz-day-night/')
    logging.info(f'Saving {base} combined ux, uy, uz day-night plot')
    fig.savefig(target + 'png-combined-uxyz-day-night/' +
                base + '.out.png-combined-uxyz-day-night.png')
    plt.close(fig)

    return


def convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath=origin):
    """Process data in BATSRUS file to create dataframe with calculated quantities.

    Inputs:
        X,Y,Z = position where magnetic field will be measured
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
        dirpath = path to directory containing base
    Outputs:
        df = dataframe containing data from BATSRUS file plus additional calculated
            parameters
        title = title to use in plots, which is derived from base (file basename)
    """

    logging.info('Parsing BATSRUS file...')

    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')

    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert(batsrus != None)

    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)

    df = pd.DataFrame()

    df['x'] = batsrus.data_arr[:, var_dict['x']][:]
    df['y'] = batsrus.data_arr[:, var_dict['y']][:]
    df['z'] = batsrus.data_arr[:, var_dict['z']][:]

    df['bx'] = batsrus.data_arr[:, var_dict['bx']][:]
    df['by'] = batsrus.data_arr[:, var_dict['by']][:]
    df['bz'] = batsrus.data_arr[:, var_dict['bz']][:]

    df['jx'] = batsrus.data_arr[:, var_dict['jx']][:]
    df['jy'] = batsrus.data_arr[:, var_dict['jy']][:]
    df['jz'] = batsrus.data_arr[:, var_dict['jz']][:]

    df['ux'] = batsrus.data_arr[:, var_dict['ux']][:]
    df['uy'] = batsrus.data_arr[:, var_dict['uy']][:]
    df['uz'] = batsrus.data_arr[:, var_dict['uz']][:]

    df['p'] = batsrus.data_arr[:, var_dict['p']][:]
    df['rho'] = batsrus.data_arr[:, var_dict['rho']][:]
    df['measure'] = batsrus.data_arr[:, var_dict['measure']][:]

    # Get the smallest cell (by volume), we will use it to normalize the
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize the magnetic field
    # to the smallest cell.
    minMeasure = df['measure'].min()

    logging.info('Calculating delta B...')

    # Calculate useful quantities
    df['r'] = ((X-df['x'])**2+(Y-df['y'])**2+(Z-df['z'])**2)**(1/2)

    # We ignore everything inside of rCurrents
    # df = df[df['r'] > rCurrents]
    df.drop(df[df['r'] < rCurrents].index)

    ##########################################################
    # Below we calculate the delta B in each cell from the
    # Biot-Savart Law.  We want the final result to be in nT.
    # dB = mu0/(4pi) (j x r)/r^3 dV
    #    = (4pi 10^(-7) [H/m])/(4pi) (10^(-6) [A/m^2]) [Re] [Re^3] / [Re^3]
    # where the fact that J is in microamps/m^2 and distances are in Re
    # is in the BATSRUS documentation.   We take Re = 6372 km = 6.371 10^6 m
    # dB = 6.371 10^(-7) [H/m][A/m^2][m]
    #    = 6.371 10^(-7) [H] [A/m^2]
    #    = 6.371 10^(-7) [kg m^2/(s^2 A^2)] [A/m^2]
    #    = 6.371 10^(-7) [kg / s^2 / A]
    #    = 6.371 10^(-7) [T]
    #    = 6.371 10^2 [nT]
    #    = 637.1 [nT] with distances in Re, j in microamps/m^2
    ##########################################################

    # Determine delta B in each cell
    df['factor'] = 637.1*df['measure']/df['r']**3
    df['dBx'] = df['factor']*(df['jy']*(Z-df['z']) - df['jz']*(Y-df['y']))
    df['dBy'] = df['factor']*(df['jz']*(X-df['x']) - df['jx']*(Z-df['z']))
    df['dBz'] = df['factor']*(df['jx']*(Y-df['y']) - df['jy']*(X-df['x']))

    # Determine magnitude of various vectors
    df['dBmag'] = np.sqrt(df['dBx']**2 + df['dBy']**2 + df['dBz']**2)
    df['jMag'] = np.sqrt(df['jx']**2 + df['jy']**2 + df['jz']**2)
    df['uMag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)

    # Normalize magnetic field, as mentioned above
    df['dBmagNorm'] = df['dBmag'] * minMeasure/df['measure']
    df['dBxNorm'] = np.abs(df['dBx'] * minMeasure/df['measure'])
    df['dByNorm'] = np.abs(df['dBy'] * minMeasure/df['measure'])
    df['dBzNorm'] = np.abs(df['dBz'] * minMeasure/df['measure'])

    logging.info('Transforming j to spherical coordinates...')

    # Transform the currents, j, into spherical coordinates

    # Determine theta and phi of the radius vector from the origin to the
    # center of the cell
    df['theta'] = np.arccos(df['z']/df['r'])
    df['phi'] = np.arctan2(df['y'], df['x'])

    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df['jr'] = df['jx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jz'] * np.cos(df['theta'])

    df['jtheta'] = df['jx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jz'] * np.sin(df['theta'])

    df['jphi'] = - df['jx'] * np.sin(df['phi']) + df['jy'] * np.cos(df['phi'])

    # Similarly, use dot-products to determine dBr, dBtheta, and dBphi
    df['dBr'] = df['dBx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['dBz'] * np.cos(df['theta'])

    df['dBtheta'] = df['dBx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['dBy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['dBz'] * np.sin(df['theta'])

    df['dBphi'] = - df['dBx'] * \
        np.sin(df['phi']) + df['dBy'] * np.cos(df['phi'])

    # Use dot product with B/BMag to get j parallel to magnetic field
    # To get j perpendicular, use j perpendicular = j - j parallel 
    df['bMag'] = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2)

    df['jparallelMag'] = (df['jx'] * df['bx'] + df['jy']
                          * df['by'] + df['jz'] * df['bz'])/df['bMag']

    df['jparallelx'] = df['jparallelMag'] * df['bx']/df['bMag']
    df['jparallely'] = df['jparallelMag'] * df['by']/df['bMag']
    df['jparallelz'] = df['jparallelMag'] * df['bz']/df['bMag']

    df['jperpendicularMag'] = np.sqrt(df['jMag']**2 - df['jparallelMag']**2)

    df['jperpendicularx'] = df['jx'] - df['jparallelx']
    df['jperpendiculary'] = df['jy'] - df['jparallely']
    df['jperpendicularz'] = df['jz'] - df['jparallelz']
    
    # Convert j perpendicular to spherical coordinates
    df['jperpendicularr'] = df['jperpendicularx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jperpendicularz'] * np.cos(df['theta'])

    df['jperpendiculartheta'] = df['jperpendicularx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jperpendicularz'] * np.sin(df['theta'])

    df['jperpendicularphi'] = - df['jperpendicularx'] * np.sin(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['phi'])

    # Divide j perpendicular into a phi piece and everything else (residual)
    df['jperpendicularphix'] = - df['jperpendicularphi'] * np.sin(df['phi'])
    df['jperpendicularphiy'] =   df['jperpendicularphi'] * np.cos(df['phi'])
    # df['jperpendicularphiy'] = 0

    df['jperpendicularphiresx'] = df['jperpendicularx'] - df['jperpendicularphix']
    df['jperpendicularphiresy'] = df['jperpendiculary'] - df['jperpendicularphiy']
    # df['jperpendicularphiresz'] = df['jperpendicularz']
    
    # Determine delta B using the parallel and perpendicular currents. They
    # should sum to the delta B calculated above for the full current, jx, jy, jz
    df['dBparallelx'] = df['factor'] * \
        (df['jparallely']*(Z-df['z']) - df['jparallelz']*(Y-df['y']))
    df['dBparallely'] = df['factor'] * \
        (df['jparallelz']*(X-df['x']) - df['jparallelx']*(Z-df['z']))
    df['dBparallelz'] = df['factor'] * \
        (df['jparallelx']*(Y-df['y']) - df['jparallely']*(X-df['x']))

    df['dBperpendicularx'] = df['factor'] * \
        (df['jperpendiculary']*(Z-df['z']) - df['jperpendicularz']*(Y-df['y']))
    df['dBperpendiculary'] = df['factor'] * \
        (df['jperpendicularz']*(X-df['x']) - df['jperpendicularx']*(Z-df['z']))
    df['dBperpendicularz'] = df['factor'] * \
        (df['jperpendicularx']*(Y-df['y']) - df['jperpendiculary']*(X-df['x']))

    # Divide the parallel currents into two - those parallel and those anti-parallel
    df['dBparax'] = deepcopy(df['dBparallelx'])
    df['dBparay'] = deepcopy(df['dBparallely'])
    df['dBparaz'] = deepcopy(df['dBparallelz'])
    df['dBparax'][df['jparallelMag'] < 0] = 0
    df['dBparay'][df['jparallelMag'] < 0] = 0
    df['dBparaz'][df['jparallelMag'] < 0] = 0

    df['dBantix'] = deepcopy(df['dBparallelx'])
    df['dBantiy'] = deepcopy(df['dBparallely'])
    df['dBantiz'] = deepcopy(df['dBparallelz'])
    df['dBantix'][df['jparallelMag'] >= 0] = 0
    df['dBantiy'][df['jparallelMag'] >= 0] = 0
    df['dBantiz'][df['jparallelMag'] >= 0] = 0
    
    # Divide delta B from perpendicular currents into two - those along phi and 
    # everything else (residual)
    df['dBperpendicularphix'] =   df['factor']*df['jperpendicularphiy']*(Z-df['z'])
    df['dBperpendicularphiy'] = - df['factor']*df['jperpendicularphix']*(Z-df['z'])
    df['dBperpendicularphiz'] = df['factor']*(df['jperpendicularphix']*(Y-df['y']) - \
                                              df['jperpendicularphiy']*(X-df['x']))
        
    df['dBperpendicularphiresx'] = df['factor']*(df['jperpendicularphiresy']*(Z-df['z']) - \
                                                 df['jperpendicularz']*(Y-df['y']))
    df['dBperpendicularphiresy'] = df['factor']*(df['jperpendicularz']*(X-df['x']) - \
                                                 df['jperpendicularphiresx']*(Z-df['z']))
    df['dBperpendicularphiresz'] = df['factor']*(df['jperpendicularphiresx']*(Y-df['y']) - \
                                                 df['jperpendicularphiresy']*(X-df['x']))

    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1] + ' (hhmmss)'

    return df, title

def create_cumulative_sum_dataframe(df):
    """Convert the dataframe with BATSRUS data to a dataframe that provides a 
    cumulative sum of dB in increasing r. To generate the cumulative sum, we 
    order the cells in terms of range r starting at the earth's center.  We 
    start the sum with a small sphere and vector sum all of the dB contributions 
    inside the sphere.  We then expand the sphere slightly and add to the sum.  
    Repeat until all cells are in the sum.

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
    Outputs:
        df_r = dataframe with all the df data plus the cumulative sums
    """

    # Sort the original data by range r, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_r = deepcopy(df)
    df_r = df_r.sort_values(by='r', ascending=True)
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBSumMag'] = np.sqrt(
        df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)

    # Do cumulative sums on currents parallel and perpendicular to the B field
    df_r['dBparallelxSum'] = df_r['dBparallelx'].cumsum()
    df_r['dBparallelySum'] = df_r['dBparallely'].cumsum()
    df_r['dBparallelzSum'] = df_r['dBparallelz'].cumsum()
    df_r['dBparallelSumMag'] = np.sqrt(df_r['dBparallelxSum']**2
                                       + df_r['dBparallelySum']**2
                                       + df_r['dBparallelzSum']**2)

    df_r['dBperpendicularxSum'] = df_r['dBperpendicularx'].cumsum()
    df_r['dBperpendicularySum'] = df_r['dBperpendiculary'].cumsum()
    df_r['dBperpendicularzSum'] = df_r['dBperpendicularz'].cumsum()
    df_r['dBperpendicularSumMag'] = np.sqrt(df_r['dBperpendicularxSum']**2
                                            + df_r['dBperpendicularySum']**2
                                            + df_r['dBperpendicularzSum']**2)

    df_r['dBperpendicularphixSum'] = df_r['dBperpendicularphix'].cumsum()
    df_r['dBperpendicularphiySum'] = df_r['dBperpendicularphiy'].cumsum()
    df_r['dBperpendicularphizSum'] = df_r['dBperpendicularphiz'].cumsum()

    df_r['dBperpendicularphiresxSum'] = df_r['dBperpendicularphiresx'].cumsum()
    df_r['dBperpendicularphiresySum'] = df_r['dBperpendicularphiresy'].cumsum()
    df_r['dBperpendicularphireszSum'] = df_r['dBperpendicularphiresz'].cumsum()

    # Do cumulative sums on para and anti currents, which are subsets of the
    # parallel sum above
    df_r['dBparaxSum'] = df_r['dBparax'].cumsum()
    df_r['dBparaySum'] = df_r['dBparay'].cumsum()
    df_r['dBparazSum'] = df_r['dBparaz'].cumsum()
    df_r['dBparaSumMag'] = np.sqrt(df_r['dBparaxSum']**2
                                   + df_r['dBparaySum']**2
                                   + df_r['dBparazSum']**2)

    df_r['dBantixSum'] = df_r['dBantix'].cumsum()
    df_r['dBantiySum'] = df_r['dBantiy'].cumsum()
    df_r['dBantizSum'] = df_r['dBantiz'].cumsum()
    df_r['dBantiSumMag'] = np.sqrt(df_r['dBantixSum']**2
                                   + df_r['dBantiySum']**2
                                   + df_r['dBantizSum']**2)

    return df_r


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

    df, title = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

    logging.info('Creating cumulative sum dB dataframe...')

    df_r = create_cumulative_sum_dataframe(df)

    # Split the data into dayside (x>=0) and nightside (x<0)
    # logging.info('Creating dayside/nightside dataframe...')

    # df_day = df[df['x'] >= 0]
    # df_night = df[df['x'] < 0]

    # Do plots...

    # logging.info('Creating dB (Norm) vs r plots...')
    # plot_db_Norm_r( df, title, base )

    # logging.info('Creating cumulative sum B vs r plots...')
    # plot_cumulative_B( df_r, title, base )

    logging.info('Creating cumulative sum B parallel/perpendicular vs r plots...')
    plot_cumulative_B_para_perp(df_r, title, base)

    # logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    # plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base )

    # logging.info('Creating day/night jx, jy, jz vs r plots...')
    # plot_jx_jy_jz_day_night( df_day, df_night, title, base )

    # logging.info('Creating jx, jy, jz vs x plots...')
    # plot_jx_jy_jz_vs_x( df, title, base)

    # logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'x')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'y')
    # plot_jr_jt_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x( df, title, base, coord = 'x')
    # plot_jp_jp_vs_x( df, title, base, coord = 'y')
    # plot_jp_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating day/night ux, uy, uz vs r plots...')
    # plot_ux_uy_uz_day_night( df_day, df_night, title, base )

    # logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    # plot_dBnorm_day_night( df_day, df_night, title, base )

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
    df2 = df_tmp.drop(df_tmp[np.logical_and(
        df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
    cutname = 'asym-jr-'
    title2 = r'asym j_r ' + title1

    if(cut_selected > 1):
        # Cut jphi vs y blob
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        cutname = 'y-jphi-' + cutname
        title2 = r'y j_{\phi} ' + title2

    if(cut_selected > 2):
        # Cut jphi vs z blob
        df2 = df2.drop(df2[np.logical_and(df2['jphi'].abs()
                       > 0.007, df2['z'].abs() < 2)].index)
        cutname = 'z-jphi-' + cutname
        title2 = r'z j_{\phi} ' + title2

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
        df2 = df_tmp.drop(df_tmp[np.logical_or(
            df_tmp['jr'].abs() <= 0.007, df_tmp['y'].abs() >= 4)].index)
        cutname = 'asym-jr-'
        title2 = r'asym $j_r$ ' + title1

    if(cut_selected == 2):
        # Cut jphi vs y blob
        df2 = df_tmp.drop(df_tmp[np.logical_and(
            df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'] <= 0.007, df2['jphi'] >= 0.03)].index)
        cutname = 'y-jphi-'
        title2 = r'y $j_{\phi}$ ' + title1

    if(cut_selected == 3):
        # Cut jphi vs z blob
        df2 = df_tmp.drop(df_tmp[np.logical_and(
            df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'].abs() <= 0.007, df2['z'].abs() >= 2)].index)
        cutname = 'z-jphi-'
        title2 = r'z $j_{\phi}$ ' + title1

    if(cut_selected == 4):
        # Cut jphi vs z blob
        df2 = df_tmp.drop(df_tmp[np.logical_and(
            df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        df2 = df2.drop(df2[np.logical_and(df2['jphi'].abs()
                       > 0.007, df2['z'].abs() < 2)].index)
        cutname = 'residual-'
        title2 = r'residual ' + title1

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
    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

    # Perform cuts on BATSRUS data
    df2, title2, cutname = perform_cuts(df1, title1, cut_selected=cut_selected)

    # Do plots...

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='x', cut=cutname)
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='y', cut=cutname)
    plot_jr_jt_jp_vs_x(df2, title2, base, coord='z', cut=cutname)

    logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    plot_jp_jp_vs_x(df2, title2, base, coord='x', cut=cutname)
    plot_jp_jp_vs_x(df2, title2, base, coord='y', cut=cutname)
    plot_jp_jp_vs_x(df2, title2, base, coord='z', cut=cutname)

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

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

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
    # # Cut all - residual
    # df5, title5, cutname5 = perform_not_cuts( df1, title1, cut_selected = 4 )
    #################################
    #################################

    logging.info('Plotting 3D extracted cuts...')

    plt.rcParams['font.size'] = 5
    vmin = 0.007
    vmax = 0.30
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.colormaps['viridis']
    # scale = 10.

    fig = plt.figure()
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    # ax.scatter(df2['x'], df2['y'], df2['z'], s=1)
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    ax = fig.add_subplot(2, 4, 2, projection='3d')
    # ax.scatter(df3['x'], df3['y'], df3['z'], s=1)
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    ax = fig.add_subplot(2, 4, 3, projection='3d')
    # ax.scatter(df4['x'], df4['y'], df4['z'], s=1)
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    ax = fig.add_subplot(2, 4, 5, projection='3d')
    # ax.quiver(df2['x'], df2['y'], df2['z'], scale*df2['jx'], scale*df2['jy'], scale*df2['jz'])
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    ax = fig.add_subplot(2, 4, 6, projection='3d')
    # ax.quiver(df3['x'], df3['y'], df3['z'],  scale*df3['jx'], scale*df3['jy'], scale*df3['jz'])
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    ax = fig.add_subplot(2, 4, 7, projection='3d')
    # ax.quiver(df4['x'], df4['y'], df4['z'],  scale*df4['jx'], scale*df4['jy'], scale*df4['jz'])
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jMag'], cmap=cmap, norm=norm)
    # plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.15, 1.15), loc=2)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'|j|', pad=0.2)

    plt.tight_layout()

    fig = plt.gcf()
    create_directory('png-3d-cuts/')
    logging.info(f'Saving {base} 3D cut plot')
    fig.savefig(target + 'png-3d-cuts/' + base + '.out.png-3d-cuts.png')
    plt.close(fig)

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
        df1 = cumulative sum for input data
        df1 - df2 = contribution due to points in asym. jr cut
        df2 - df3 = contribution due to points in y jphi cut
        df3 - df4 = contribution due to points in z jphi cut
    """

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

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

    logging.info(
        'Calculate cumulative sums for dataframes for extracted cuts...')

    df1 = create_cumulative_sum_dataframe(df1)
    df2 = create_cumulative_sum_dataframe(df2)
    df3 = create_cumulative_sum_dataframe(df3)
    df4 = create_cumulative_sum_dataframe(df4)

    return df1['dBzSum'].iloc[-1], \
        df1['dBparazSum'].iloc[-1], \
        df1['dBantizSum'].iloc[-1], \
        df1['dBparallelzSum'].iloc[-1], \
        df1['dBperpendicularzSum'].iloc[-1], \
        df1['dBperpendicularphizSum'].iloc[-1], \
        df1['dBperpendicularphireszSum'].iloc[-1], \
        df1['dBzSum'].iloc[-1] - df2['dBzSum'].iloc[-1], \
        df1['dBparazSum'].iloc[-1] - df2['dBparazSum'].iloc[-1], \
        df1['dBantizSum'].iloc[-1] - df2['dBantizSum'].iloc[-1], \
        df1['dBparallelzSum'].iloc[-1] - df2['dBparallelzSum'].iloc[-1], \
        df1['dBperpendicularzSum'].iloc[-1] - df2['dBperpendicularzSum'].iloc[-1], \
        df1['dBperpendicularphizSum'].iloc[-1] - df2['dBperpendicularphizSum'].iloc[-1], \
        df1['dBperpendicularphireszSum'].iloc[-1] - df2['dBperpendicularphireszSum'].iloc[-1], \
        df2['dBzSum'].iloc[-1] - df3['dBzSum'].iloc[-1], \
        df2['dBparazSum'].iloc[-1] - df3['dBparazSum'].iloc[-1], \
        df2['dBantizSum'].iloc[-1] - df3['dBantizSum'].iloc[-1], \
        df2['dBparallelzSum'].iloc[-1] - df3['dBparallelzSum'].iloc[-1], \
        df2['dBperpendicularzSum'].iloc[-1] - df3['dBperpendicularzSum'].iloc[-1], \
        df2['dBperpendicularphizSum'].iloc[-1] - df3['dBperpendicularphizSum'].iloc[-1], \
        df2['dBperpendicularphireszSum'].iloc[-1] - df3['dBperpendicularphireszSum'].iloc[-1], \
        df3['dBzSum'].iloc[-1] - df4['dBzSum'].iloc[-1], \
        df3['dBparazSum'].iloc[-1] - df4['dBparazSum'].iloc[-1], \
        df3['dBantizSum'].iloc[-1] - df4['dBantizSum'].iloc[-1], \
        df3['dBparallelzSum'].iloc[-1] - df4['dBparallelzSum'].iloc[-1], \
        df3['dBperpendicularzSum'].iloc[-1] - df4['dBperpendicularzSum'].iloc[-1], \
        df3['dBperpendicularphizSum'].iloc[-1] - df4['dBperpendicularphizSum'].iloc[-1], \
        df3['dBperpendicularphireszSum'].iloc[-1] - df4['dBperpendicularphireszSum'].iloc[-1], \
        df4['dBzSum'].iloc[-1], \
        df4['dBparazSum'].iloc[-1], \
        df4['dBantizSum'].iloc[-1], \
        df4['dBparallelzSum'].iloc[-1], \
        df4['dBperpendicularzSum'].iloc[-1], \
        df4['dBperpendicularphizSum'].iloc[-1], \
        df4['dBperpendicularphireszSum'].iloc[-1]


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
    plt.rcParams["figure.figsize"] = [3.6, 3.2]

    n = len(files)

    b_original = [None] * n
    b_original_para = [None] * n
    b_original_anti = [None] * n
    b_original_parallel = [None] * n
    b_original_perp = [None] * n
    b_original_perpphi = [None] * n
    b_original_perpphires = [None] * n
    b_asym_jr = [None] * n
    b_asym_jr_para = [None] * n
    b_asym_jr_anti = [None] * n
    b_asym_jr_parallel = [None] * n
    b_asym_jr_perp = [None] * n
    b_asym_jr_perpphi = [None] * n
    b_asym_jr_perpphires = [None] * n
    b_y_jphi = [None] * n
    b_y_jphi_para = [None] * n
    b_y_jphi_anti = [None] * n
    b_y_jphi_parallel = [None] * n
    b_y_jphi_perp = [None] * n
    b_y_jphi_perpphi = [None] * n
    b_y_jphi_perpphires = [None] * n
    b_z_jphi = [None] * n
    b_z_jphi_para = [None] * n
    b_z_jphi_anti = [None] * n
    b_z_jphi_parallel = [None] * n
    b_z_jphi_perp = [None] * n
    b_z_jphi_perpphi = [None] * n
    b_z_jphi_perpphires = [None] * n
    b_residual = [None] * n
    b_residual_para = [None] * n
    b_residual_anti = [None] * n
    b_residual_parallel = [None] * n
    b_residual_perp = [None] * n
    b_residual_perpphi = [None] * n
    b_residual_perpphires = [None] * n
    b_times = [None] * n
    b_index = [None] * n

    for i in range(n):
        # for i in range(4):
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
        # main components of the field - the complete field, para and anti are subsets
        # of parallel component, parallel (aka para+anti), and perpendicular
        #
        b_original[i], b_original_para[i], b_original_anti[i], b_original_parallel[i], b_original_perp[i], \
            b_original_perpphi[i], b_original_perpphires[i], \
            b_asym_jr[i], b_asym_jr_para[i], b_asym_jr_anti[i], b_asym_jr_parallel[i], b_asym_jr_perp[i], \
            b_asym_jr_perpphi[i], b_asym_jr_perpphires[i], \
            b_y_jphi[i], b_y_jphi_para[i], b_y_jphi_anti[i], b_y_jphi_parallel[i], b_y_jphi_perp[i], \
            b_y_jphi_perpphi[i], b_y_jphi_perpphires[i], \
            b_z_jphi[i], b_z_jphi_para[i], b_z_jphi_anti[i], b_z_jphi_parallel[i], b_z_jphi_perp[i], \
            b_z_jphi_perpphi[i], b_z_jphi_perpphires[i], \
            b_residual[i], b_residual_para[i], b_residual_anti[i], b_residual_parallel[i], b_residual_perp[i], \
            b_residual_perpphi[i], b_residual_perpphires[i] = \
            process_sum_db_with_cuts(X, Y, Z, base=files[i])

    plt.figure()
    plt.plot(b_times, b_original, ls='solid', color='black')
    plt.plot(b_times, b_asym_jr, ls='dashed', color='blue')
    plt.plot(b_times, b_y_jphi,  ls='dashdot', color='blue')
    plt.plot(b_times, b_z_jphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_residual, ls='solid', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend(['Original', r'Asym $j_r$ only', r'y $j_\phi$ only',
               r'z $j_\phi$ only', r'Residual'])

    plt.figure()
    plt.plot(b_times, b_original, ls='solid', color='black')
    plt.plot(b_times, b_original_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_original_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_original_perpphires,  ls='dotdash', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend(['Original', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residue'])

    plt.figure()
    plt.plot(b_times, b_asym_jr, ls='solid', color='black')
    plt.plot(b_times, b_asym_jr_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_asym_jr_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_asym_jr_perpphires,  ls='dotdash', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend([r'Asymmetric $j_r$', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residue'])

    plt.figure()
    plt.plot(b_times, b_y_jphi, ls='solid', color='black')
    plt.plot(b_times, b_y_jphi_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_y_jphi_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_y_jphi_perpphires,  ls='dotdash', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend([r'y $j_{\phi}$', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residue'])

    plt.figure()
    plt.plot(b_times, b_z_jphi, ls='solid', color='black')
    plt.plot(b_times, b_z_jphi_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_z_jphi_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_z_jphi_perpphires,  ls='dotdash', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend([r'z $j_{\phi}$', r'Perpendicular $\phi$', r'Perpendicular Residue'])

    plt.figure()
    plt.plot(b_times, b_residual, ls='solid', color='black')
    plt.plot(b_times, b_residual_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_residual_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_residual_perpphires,  ls='dotdash', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(-dB_sum_limits[1], dB_sum_limits[1])
    plt.legend([r'Residual', r'Perpendicular $\phi$', r'Perpendicular Residue'])

    plt.figure()
    plt.plot(b_original_para, b_original_anti, ls='solid', color='blue')
    plt.xlabel(r'Original $B_z$ Parallel')
    plt.ylabel(r'Original $B_z$ Anti-Parallel')

    plt.figure()
    plt.plot(b_asym_jr_para, b_asym_jr_anti, ls='solid', color='blue')
    plt.xlabel(r'Asym $j_r$ $B_z$ Parallel')
    plt.ylabel(r'Asym $j_r$ $B_z$ Anti-Parallel')

    plt.figure()
    plt.plot(b_y_jphi_para, b_y_jphi_anti, ls='solid', color='blue')
    plt.xlabel(r'y $j_\phi$ $B_z$ Parallel')
    plt.ylabel(r'y $j_\phi$ $B_z$ Anti-Parallel')

    plt.figure()
    plt.plot(b_z_jphi_para, b_z_jphi_anti, ls='solid', color='blue')
    plt.xlabel(r'z $j_\phi$ $B_z$ Parallel')
    plt.ylabel(r'z $j_\phi$ $B_z$ Anti-Parallel')

    plt.figure()
    plt.plot(b_residual_para, b_residual_anti, ls='solid', color='blue')
    plt.xlabel(r'Residual $B_z$ Parallel')
    plt.ylabel(r'Residual $B_z$ Anti-Parallel')

    return


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

    l = glob.glob(base + '.out')

    # Strip off extension
    for i in range(len(l)):
        l[i] = (l[i].split('.'))[0]

    l.sort()

    return l


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

    l1.sort()

    return l1


if __name__ == "__main__":
    files = get_files()
    # files = get_files_unconverted( tgtsubdir = 'png-combined-sum-dB-para-perp-comp-r/' )

    # logging.info('Num. of files: ' + str(len(files)))

    X = 1
    Y = 0
    Z = 0

    # for i in range(len(files)):
    # for i in range(1):
        # y,n,d,h,m,s = date_time(files[i])
        # n_geo, e_geo, z_geo = nez((y,n,d,h,m,s), ((X,Y,Z), (X,Y,Z), (X,Y,Z)), 'GSM')
        # print(n_geo, e_geo, z_geo)

        # process_data(X, Y, Z, base = files[i])

        # process_data_with_cuts(X, Y, Z, base = files[i], cut_selected = 3)

        # if(i>1): process_3d_cut_plots(X, Y, Z, base = files[i])

    loop_sum_db_thru_cuts(X, Y, Z, files)


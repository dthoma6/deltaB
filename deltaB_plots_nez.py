#!/usr/bin/env python3.
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 09:42:06 2022

@author: dean
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
import swmfio
from os import makedirs
from os.path import exists
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

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

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.8, 7.2]
# plt.rcParams["figure.figsize"] = [3.6,3.2]
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 300
plt.rcParams['font.size'] = 7
plt.rcParams['axes.grid'] = True

# Create namedtuple used to draw multiple plots
#   df = dataframe with data quick
#   x, y = names of columns in dataframe to be plotted
#   logx, logy = Boolean, use log scale in plot
#   xlabel, ylabel = labels for axes
#   xlim, ylim = limits of axes
#   title = plot title
plotargs = namedtuple('plotargs', ['df',
                                   'x', 'y',
                                   'logx', 'logy',
                                   'xlabel', 'ylabel',
                                   'xlim', 'ylim',
                                   'title'])

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

def create_directory( folder ):
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

# def plot_2x4(base, suffix, plots, cols=4, rows=2 ):
#     """Plot 2 rows of 4 plots each (8 plots total).  

#     Inputs:
#         tgt = .
#         base = basename of file used to create file name for plot.  base is
#             derived from file with BATSRUS data.
#         suffix = suffix is used to generate file names and subdirectory.
#             Plots are saved in target + suffix directory, target is the 
#             overarching directory for all plots.  It contains subdirectories
#             (suffix) where different types of plots are saved
#         plots = a list of plotargs namedtuples that have the plot parameters
#     Outputs:
#         None - other than the figures that are saved to files
#      """
#     assert( len(plots) > 0 and len(plots) <= cols*rows )
    
#     for i in range(len(plots)):
#        if( plots[i] != None): 
#            plots[i].df.plot.scatter(x=plots[i].x, y=plots[i].y, 
#                                     ax = plt.subplot(rows,cols,i+1),
#                                     logx=plots[i].logx, 
#                                     logy=plots[i].logy, 
#                                     xlim=plots[i].xlim, 
#                                     ylim=plots[i].ylim,
#                                     xlabel=plots[i].xlabel, 
#                                     ylabel=plots[i].ylabel, 
#                                     title=plots[i].title, 
#                                     s=1)


#     fig = plt.gcf()
#     create_directory(suffix +'/')
#     logging.info(f'Saving {base} {suffix} plot')
#     fig.savefig(target + suffix + '/' + base + '.out.' + suffix + '.png')
#     plt.close(fig)

#     return

from vtkmodules.vtkChartsCore import (
    vtkAxis,
    vtkChart,
    vtkChartXY,
    vtkPlotPoints
)
from vtk import vtkPen
from vtkmodules.vtkCommonColor import vtkNamedColors
# from vtkmodules.vtkCommonCore import vtkFloatArray
from vtkmodules.vtkCommonDataModel import vtkTable
from vtkmodules.vtkRenderingContext2D import (
    vtkContextActor,
    vtkContextScene
)
from vtkmodules.vtkRenderingCore import (
    vtkRenderWindow,
    # vtkRenderWindowInteractor,
    vtkRenderer
)
from vtk.util import numpy_support as ns
from vtk import vtkRenderLargeImage, vtkPNGWriter

def plot_NxM(base, suffix, plots, cols=4, rows=2, plottype='scatter' ):
    """Plot 2 rows of 4 plots each (8 plots total).  

    Inputs:
        tgt = .
        base = basename of file used to create file name for plot.  base is
            derived from file with BATSRUS data.
        suffix = suffix is used to generate file names and subdirectory.
            Plots are saved in target + suffix directory, target is the 
            overarching directory for all plots.  It contains subdirectories
            (suffix) where different types of plots are saved
        plots = a list of plotargs namedtuples that have the plot parameters
    Outputs:
        None - other than the figures that are saved to files
     """
    assert( len(plots) > 0 and len(plots) <= cols*rows )
    assert( plottype == 'scatter' or plottype == 'line' )

    colors = vtkNamedColors()

    renwin = vtkRenderWindow()
    renwin.SetWindowName('MultiplePlots')
    renwin.OffScreenRenderingOn()

    # Setup the viewports
    renderer_sz_x = 600 * cols
    renderer_sz_y = 600 * rows
    renwin.SetSize(renderer_sz_x, renderer_sz_y)

    # Set up viewports, each plot in plots is drawn in a different viewpot
    # The goal is to have a set of side-by-side plots in rows and columns
    viewports = list()
    for row in range(rows):
        for col in range(cols):
            viewports.append([float(col) / cols,
                              float(rows - (row + 1)) / rows,
                              float(col + 1) / cols,
                              float(rows - row) / rows])    

    # Link the renderers to the viewports and create the charts
    renderers = list()
    charts = list()
    scenes = list()
    actors = list()
    tables = list()
    x_axes = list()
    y_axes = list()
    x_arrays = list()
    y_arrays = list()
    points = list()
    
    for i in  range( len(plots) ):
        if( plots[i] != None ):
            # Create renderer for each chart
            renderers.append( vtkRenderer() )
            renderers[-1].SetBackground( colors.GetColor3d('White') )
            renderers[-1].SetViewport( viewports[i] )
            renwin.AddRenderer( renderers[-1] )
    
            # Create chart along with the scene and actor for it.  THe generic 
            # flow is chart -> scene -> actor -> renderer -> renderwindow
            charts.append( vtkChartXY() )
            scenes.append( vtkContextScene() )
            actors.append( vtkContextActor() )
        
            scenes[-1].AddItem(charts[-1])
            actors[-1].SetScene(scenes[-1])
            renderers[-1].AddActor(actors[-1])
            scenes[-1].SetRenderer(renderers[-1])
            
            # Set up characteristics of x and y axes - color, the titles, and
            # that we will fix the range.  The actual value for the ranges will
            # be set below.
            x_axes.append( charts[-1].GetAxis( vtkAxis.BOTTOM ) )
            x_axes[-1].GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            x_axes[-1].SetTitle( plots[i].xlabel )
            x_axes[-1].SetBehavior(vtkAxis.FIXED)
             
            y_axes.append( charts[-1].GetAxis(vtkAxis.LEFT) )
            y_axes[-1].GetGridPen().SetColor( colors.GetColor4ub( "LightGrey" ) )
            y_axes[-1].SetTitle( plots[i].ylabel )
            y_axes[-1].SetBehavior(vtkAxis.FIXED)
            
            # Set title for chart
            charts[-1].SetTitle( plots[i].title )
    
            # Store the data to be plotted in a vtkTable, the data is taken 
            # from the dataframe in plots, and will be graphed in the associated chart
            tables.append( vtkTable() ) 
            
            x_arrays.append( ns.numpy_to_vtk( (plots[i].df)[ plots[i].x ].to_numpy() ) )
            x_arrays[-1].SetName( plots[i].xlabel )
            tables[-1].AddColumn( x_arrays[-1] )
  
            y_arrays.append( ns.numpy_to_vtk( (plots[i].df)[ plots[i].y ].to_numpy() ) )
            y_arrays[-1].SetName( plots[i].ylabel )
            tables[-1].AddColumn( y_arrays[-1] )
            
            # As appropriate, use log scales and set the min/max values for the
            # axes
            if( plots[i].logx ):
                x_axes[-1].LogScaleOn()
            x_axes[-1].SetMinimum( plots[i].xlim[0] )
            x_axes[-1].SetMaximum( plots[i].xlim[1] )

            if( plots[i].logy ):
                y_axes[-1].LogScaleOn()
            y_axes[-1].SetMinimum( plots[i].ylim[0] )
            y_axes[-1].SetMaximum( plots[i].ylim[1] )

            # Either plot a scatter plot or a line graph
            if plottype == 'scatter':
                points.append( charts[-1].AddPlot( vtkChart.POINTS ) )
                points[-1].SetMarkerStyle(vtkPlotPoints.CIRCLE)
            else:  
                points.append( charts[-1].AddPlot( vtkChart.LINE ) )
                points[-1].GetPen().SetLineType( vtkPen.SOLID_LINE )
            points[-1].SetInputData(tables[-1], 0, 1)
            points[-1].SetColor(*colors.GetColor4ub('Blue'))
            points[-1].SetWidth(1.0)
            points[-1].SetMarkerSize(0.1)
    
    # Now that the charts are set up, render the graphs
    renwin.Render()

    # Store the graphic in a file.
    create_directory(suffix +'/')
    logging.info(f'Saving {base} {suffix} plot')
    fn=target + suffix + '/' + base + '.out.' + suffix + '.png'
    
    renLgeIm = vtkRenderLargeImage()
    imgWriter = vtkPNGWriter()
    renLgeIm.SetInput( renwin.GetRenderers().GetFirstRenderer() )
    renLgeIm.SetMagnification(1)
    imgWriter.SetInputConnection( renLgeIm.GetOutputPort() )
    imgWriter.SetFileName(fn)
    imgWriter.Write()
    
    return

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

    plot_NxM(base, 'png-dBNorm-r', plots, cols=4, rows=1 )
    
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

    plot_NxM(base, 'png-dBNorm-various-day-night', plots )
    
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
    
    plot_NxM(base, 'png-sum-dB-r', plots, cols=4, rows=1 )

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

    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularxSum'], s=1, color='blue')
    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularphixSum'], s=1, color='red')
    plt.subplot(2, 4, 5).scatter(x=df_r['r'], y=df_r['dBperpendicularphiresxSum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_x (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBx_sum_limits[0], dBx_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'])

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

    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularySum'], s=1, color='blue')
    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularphiySum'], s=1, color='red')
    plt.subplot(2, 4, 6).scatter(x=df_r['r'], y=df_r['dBperpendicularphiresySum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_y (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBy_sum_limits[0], dBy_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'])


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

    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularzSum'], s=1, color='blue')
    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularphizSum'], s=1, color='red')
    plt.subplot(2, 4, 7).scatter(x=df_r['r'], y=df_r['dBperpendicularphireszSum'], s=1, color='green')
    plt.title(r'$\perp$ ' + title)
    plt.xscale("log")
    plt.xlabel(r'$r/R_E$')
    plt.ylabel(r'$\Sigma_r \delta B_z (j_{\perp})$')
    plt.xlim(1, 1000)
    plt.ylim(dBz_sum_limits[0], dBz_sum_limits[1])
    plt.legend([r'$\perp_{tot}$', r'$\perp_\phi$', r'$\perp_{residual}$'])


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
    create_directory('png-sum-dB-para-perp-comp-r/')
    logging.info(f'Saving {base} dB parallel/perpendicular plot')
    fig.savefig(target + 'png-sum-dB-para-perp-comp-r/' +
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

    plot_NxM(base, 'png-various-r-day-night', plots )
    
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

    plot_NxM(base, 'png-jxyz-r-day-night', plots )

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

    plot_NxM(base, 'png-uxyz-r-day-night', plots )

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
    
    plot_NxM(base, 'png-jrtp-'+cut+coord, plots )

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
    
    plot_NxM(base, 'png-jpp-'+cut+coord, plots, cols=3, rows=2 )

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
    
    plot_NxM(base, 'png-jrtp-cdf', plots, cols=3, rows=1 )

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
    # df['jperpendicularphiz'] = 0

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

    return df_r

def create_jrtp_cdfs(df):
    """Use the dataframe with BATSRUS dataframe to develop jr, jtheta, jphi
    CDFs.

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities
    Outputs:
        cdf_jr, cdf_jtheta, cdf_jphi = jr, jtheta, jphi CDFs.
    """

    # Sort the original data, ascending
    # Then calculate the total B based upon summing the vector dB values
    # starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_jr = deepcopy(df)
    df_jtheta = deepcopy(df)
    df_jphi = deepcopy(df)
    
    df_jr = df_jr.sort_values(by='jr', ascending=True)
    df_jr['cdfIndex'] = np.arange(1, len(df_jr)+1)/float(len(df_jr))

    df_jtheta = df_jtheta.sort_values(by='jtheta', ascending=True)
    df_jtheta['cdfIndex'] = np.arange(1, len(df_jtheta)+1)/float(len(df_jtheta))

    df_jphi = df_jphi.sort_values(by='jphi', ascending=True)
    df_jphi['cdfIndex'] = np.arange(1, len(df_jphi)+1)/float(len(df_jphi))

    return df_jr, df_jtheta, df_jphi

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

    logging.info('Creating dayside/nightside dataframe...')

    df_day = df[df['x'] >= 0]
    df_night = df[df['x'] < 0]

    # Do plots...

    logging.info('Creating dB (Norm) vs r plots...')
    plot_db_Norm_r( df, title, base )
    
    logging.info('Creating day/night dB (Norm) vs rho, p, etc. plots...')
    plot_dBnorm_various_day_night( df_day, df_night, title, base )
    
    logging.info('Creating cumulative sum B vs r plots...')
    plot_sum_dB( df_r, title, base )

    logging.info('Creating cumulative sum B parallel/perpendicular vs r plots...')
    plot_cumulative_B_para_perp(df_r, title, base)

    logging.info('Creating day/night rho, p, jMag, uMag vs r plots...')
    plot_rho_p_jMag_uMag_day_night( df_day, df_night, title, base )

    logging.info('Creating day /night jx, jy, jz vs r plots...')
    plot_jx_jy_jz_day_night( df_day, df_night, title, base )

    logging.info('Creating day/night ux, uy, uz vs r plots...')
    plot_ux_uy_uz_day_night( df_day, df_night, title, base )

    logging.info('Creating jr, jtheta, jphi vs x,y,z plots...')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'x')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'y')
    plot_jr_jt_jp_vs_x( df, title, base, coord = 'z')

    # logging.info('Creating jparallel and jperpendicular vs x,y,z plots...')
    # plot_jp_jp_vs_x( df, title, base, coord = 'x')
    # plot_jp_jp_vs_x( df, title, base, coord = 'y')
    # plot_jp_jp_vs_x( df, title, base, coord = 'z')

    logging.info('Creating jrtp CDFs...')
    df_jr, df_jtheta, df_jphi = create_jrtp_cdfs(df)
    plot_jrtp_cdfs(df_jr, df_jtheta, df_jphi, title, base)

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
    # df2 = df_tmp.drop(df_tmp[np.logical_and( 
#        df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
    cutname = r'jr-'
    title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected > 1):
        # Cut jphi vs y blob
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
#            df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        cutname = 'jphifar-' + cutname
        title2 = r'$j_\phi$ Peaks (far) ' + title2

    if(cut_selected > 2):
        # Cut jphi vs z blob
        df2 = df2.drop(df2[df2['jphi'].abs() > cut3_jphimin].index)
#            df2[np.logical_and(df2['jphi'].abs() > 0.007, df2['z'].abs() < 2)].index)
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
        # df2 = df_tmp.drop(df_tmp[np.logical_or(
        #     df_tmp['jr'].abs() <= 0.007, df_tmp['y'].abs() >= 4)].index)
        cutname = 'jr-'
        title2 = r'$j_r$ Peaks ' + title1

    if(cut_selected == 2):
        # Cut jphi vs y blob
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        # df2 = df_tmp.drop(df_tmp[np.logical_and(
        #     df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        df2 = df2.drop(
            df2[np.logical_or(df2['jphi'].abs() <= cut2_jphimin, df2['r'] <= cut2_rmin)].index)
        # df2 = df2.drop(
        #     df2[np.logical_or(df2['jphi'] <= 0.007, df2['jphi'] >= 0.03)].index)
        cutname = 'jphifar-'
        title2 = r'$j_\phi$ Peaks (far) ' + title1

    if(cut_selected == 3):
        # Cut jphi vs z blob
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        # df2 = df_tmp.drop(df_tmp[np.logical_and(
        #     df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
        # df2 = df2.drop(
            # df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        # df2 = df2.drop(
            # df2[np.logical_or(df2['jphi'].abs() <= 0.007, df2['z'].abs() >= 2)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() <= cut3_jphimin].index)
        cutname = 'jphinear-'
        title2 = r'$j_\phi$ (near) ' + title1

    if(cut_selected == 4):
        df2 = df_tmp.drop(df_tmp[df_tmp['jr'].abs() > cut1_jrmin].index)
        df2 = df2.drop(
            df2[np.logical_and(df2['jphi'].abs() > cut2_jphimin, df2['r'] > cut2_rmin)].index)
        df2 = df2.drop(df2[df2['jphi'].abs() > cut3_jphimin].index)
        # df2 = df_tmp.drop(df_tmp[np.logical_and(
        #     df_tmp['jr'].abs() > 0.007, df_tmp['y'].abs() < 4)].index)
        # df2 = df2.drop(
        #     df2[np.logical_and(df2['jphi'] > 0.007, df2['jphi'] < 0.03)].index)
        # df2 = df2.drop(df2[np.logical_and(df2['jphi'].abs()
        #                > 0.007, df2['z'].abs() < 2)].index)
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
    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

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

    logging.info('Plotting 3D extracted cuts...')

    plt.rcParams['font.size'] = 5
    from matplotlib.colors import LogNorm, CenteredNorm, SymLogNorm
    # norm = LogNorm(vmin=vmin, vmax=vmax)
    # norm = CenteredNorm(vcenter=0, halfrange=vmax)
    norm = SymLogNorm(linthresh=vmin, vmin=-vmax, vmax=vmax)
    # cmap = plt.colormaps['viridis']
    cmap = plt.colormaps['coolwarm']

    fig = plt.figure()
    ax = fig.add_subplot(2, 4, 1, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 2, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 3, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 5, projection='3d')
    sc = ax.scatter(df2['x'], df2['y'], df2['z'], s=1,
                    c=df2['jr'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title2)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{r} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 6, projection='3d')
    sc = ax.scatter(df3['x'], df3['y'], df3['z'], s=1,
                    c=df3['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title3)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

    ax = fig.add_subplot(2, 4, 7, projection='3d')
    sc = ax.scatter(df4['x'], df4['y'], df4['z'], s=1,
                    c=df4['jphi'], cmap=cmap, norm=norm)
    ax.set_xlabel(r'$x/R_e$')
    ax.set_ylabel(r'$y/R_e$')
    ax.set_zlabel(r'$z/R_e$')
    ax.set_title(title4)
    ax.set_xlim(plot3d_limits[0], plot3d_limits[1])
    ax.set_ylim(plot3d_limits[0], plot3d_limits[1])
    ax.set_zlim(plot3d_limits[0], plot3d_limits[1])
    ax.view_init(-140, 60)
    fig.colorbar(sc, shrink=0.4, location='right', label=r'$ j_{\phi} $', pad=0.2)

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
        df1 = cumulative sum for original (all) data in north-east-zenith
        df1 - df2 = contribution due to points in asym. jr cut in nez
        df2 - df3 = contribution due to points in y jphi cut in nez
        df3 - df4 = contribution due to points in z jphi cut in nez
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

    df1, title1 = convert_BATSRUS_to_dataframe(X, Y, Z, base, dirpath)

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
    plt.rcParams["figure.figsize"] = [3.6, 3.2]

    n = len(files)

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

    plt.figure()
    plt.plot(b_times, b_original, ls='solid', color='black')
    plt.plot(b_times, b_asym_jr, ls='dashed', color='blue')
    plt.plot(b_times, b_y_jphi,  ls='dashdot', color='blue')
    plt.plot(b_times, b_z_jphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_residual, ls='solid', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend(['Original', r'$j_r$ Peaks', r'$j_\phi$ Peaks (far)', r'$j_\phi$ Peaks (near)', r'Residual'])

    plt.figure()
    plt.plot(b_times, b_original, ls='solid', color='black')
    plt.plot(b_times, b_original_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_original_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_original_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend(['Original', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    plt.figure()
    plt.plot(b_times, b_asym_jr, ls='solid', color='black')
    plt.plot(b_times, b_asym_jr_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_asym_jr_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_asym_jr_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend([r'$j_r$ Peaks', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    plt.figure()
    plt.plot(b_times, b_y_jphi, ls='solid', color='black')
    plt.plot(b_times, b_y_jphi_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_y_jphi_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_y_jphi_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend([r'$j_\phi$ Peaks (far)', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    plt.figure()
    plt.plot(b_times, b_z_jphi, ls='solid', color='black')
    plt.plot(b_times, b_z_jphi_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_z_jphi_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_z_jphi_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend([r'$j_\phi$ Peaks (near)', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    plt.figure()
    plt.plot(b_times, b_residual, ls='solid', color='black')
    plt.plot(b_times, b_residual_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_residual_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_residual_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend([r'Residual', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

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
    plt.rcParams["figure.figsize"] = [3.6, 3.2]

    n = len(files)

    b_original = [None] * n
    b_original_parallel = [None] * n
    b_original_perp = [None] * n
    b_original_perpphi = [None] * n
    b_original_perpphires = [None] * n
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
        # main components of the field - the complete field, parallel and perpendicular
        # (perpphi and perpphires are components of perpendicular)
        b_original[i], b_original_parallel[i], b_original_perp[i], \
            b_original_perpphi[i], b_original_perpphires[i] = \
            process_sum_db(X, Y, Z, base=files[i])

    plt.figure()
    plt.plot(b_times, b_original, ls='solid', color='black')
    plt.plot(b_times, b_original_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_original_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_original_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Total $B_z$ at (1,0,0)')
    plt.ylim(dB_sum_limits2[0], dB_sum_limits2[1])
    plt.legend(['Total', r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    b_fraction_parallel = [m/n for m, n in zip(b_original_parallel, b_original)]
    b_fraction_perpphi = [m/n for m, n in zip(b_original_perpphi, b_original)]
    b_fraction_perpphires = [m/n for m, n in zip(b_original_perpphires, b_original)]
    
    plt.figure()
    plt.plot(b_times, b_fraction_parallel, ls='dashed', color='blue')
    plt.plot(b_times, b_fraction_perpphi,  ls='dotted', color='blue')
    plt.plot(b_times, b_fraction_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Fraction of Total $B_z$ at (1,0,0)')
    plt.ylim(-0.5,1.5)
    plt.legend([r'Parallel', r'Perpendicular $\phi$', r'Perpendicular Residual'])

    plt.figure()
    plt.plot(b_times, b_fraction_parallel, ls='dashed', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Fraction of Total $B_z$ at (1,0,0)')
    plt.ylim(-0.5,1.5)
    plt.legend([r'Parallel'])

    plt.figure()
    plt.plot(b_times, b_fraction_perpphi,  ls='dotted', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Fraction of Total $B_z$ at (1,0,0)')
    plt.ylim(-0.5,1.5)
    plt.legend([r'Perpendicular $\phi$'])

    plt.figure()
    plt.plot(b_times, b_fraction_perpphires,  ls='dashdot', color='blue')
    plt.xlabel(r'Time (hr)')
    # plt.ylabel(r'Total $B_N$ at (1,0,0)')
    plt.ylabel(r'Fraction of Total $B_z$ at (1,0,0)')
    plt.ylim(-0.5,1.5)
    plt.legend([r'Perpendicular Residual'])

    return

if __name__ == "__main__":
    # if COLABA:
    #     files = get_files(base='3d__var_2_e*')
    # else:
    #     files = get_files(base='3d__*')
        
    if COLABA:
        files = get_files_unconverted( tgtsubdir = 'png-jrtp-cdf/', base='3d__var_2_e*' )
    else:
        files = get_files_unconverted( tgtsubdir = 'png-jrtp-cdf/', base='3d__*' )

    logging.info('Num. of files: ' + str(len(files)))

    X = 1
    Y = 0
    Z = 0

    for i in range(len(files)):
    # for i in range(1):
        process_data(X, Y, Z, files[i])
        # process_data_with_cuts(X, Y, Z, files[i], cut_selected = 3)
        # if(i>1): process_3d_cut_plots(X, Y, Z, files[i])
        # process_3d_cut_plots(X, Y, Z, files[i])

    # loop_sum_db_thru_cuts(X, Y, Z, files)
    # loop_sum_db(X, Y, Z, files)
    
    # time = "2001-01-01T12:00:00"
    # n_geo, e_geo, z_geo = nez(time, (1,0,0), 'GSM')
    # print( n_geo, e_geo, z_geo)
    
    # for i in range(len(files)):
        # y,n,d,h,m,s = date_time(files[i])
        # n_geo, e_geo, z_geo = nez((y,n,d,h,m,s), ((X,Y,Z), (X,Y,Z), (X,Y,Z)), 'GSM')
        # print(n_geo, e_geo, z_geo)




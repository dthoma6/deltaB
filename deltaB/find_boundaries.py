#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

"""
This script identifies points on the bow shock, magnetopause, and neutral
sheet.  The bow shock is when there is a dramatic reduction in the solar wind 
velocity.  The magnetopause is when magnetic pressure (B^2//2/mu0) equals the 
dynamic pressure (rho u^2).  (Note, this dynamic pressure is also known as 
momentum flux or the ram pressure, and is different from the dynamic 
pressure = 1/2 rho u^2).  The neutral sheet is at the center of the plasmasheet 
in the tail of the magnetosphere, where the magnetic field (Bx) switches directions.
""" 

import os.path
import logging
import swmfio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.interpolate import LinearNDInterpolator

from deltaB import create_directory, sqwireframe, pointcloud

# Physical constant mu0
MU0 = 1.25663706212*10**-6
    
# We assume that polytropic index, gamma = 5/3. This is the polytropic
# index of an ideal monatomic gas
GAMMA = 5/3

# In dynamic (ram) pressure equation, represents efficiency coefficient
# KAPPA = 1 is specular reflection.
KAPPA = 1

# Fit Shue eqn to smooth magnetopause normals or use numeric gradient of fit to find normals
USE_SHUE = True

# Use one parameter fit for bow shock
USE_ONE_PARAMETER = False
# Use two parameter fit for bow shock
USE_TWO_PARAMETER = False
# Use general quadratic conic section fit for bow shock
# a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z + k = 0 
# and we divide by k and redefine coeficients to get
# => a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z + 1 = 0
USE_GEN_PARAMETER = True
# One and only one of the flags can be true
assert( USE_ONE_PARAMETER + USE_TWO_PARAMETER + USE_GEN_PARAMETER == 1 )

if USE_GEN_PARAMETER:
    # Use scipy fsolve to solve general eqn for x
    from scipy.optimize import fsolve

logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

# info = {...} example is below

# data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

# info = {
#         "model": "SWMF",
#         "run_name": "Bob_Weigel_031023_1",
#         "rCurrents": 4.0,
#         "rIonosphere": 1.01725,
#         "file_type": "cdf",
#         "dir_run": os.path.join(data_dir, "Bob_Weigel_031023_1"),
#         "dir_plots": os.path.join(data_dir, "Bob_Weigel_031023_1.plots"),
#         "dir_derived": os.path.join(data_dir, "Bob_Weigel_031023_1.derived"),
#         "dir_ionosphere": os.path.join(data_dir, "Bob_Weigel_031023_1/IONO-2D_CDF"),
#         "dir_magnetosphere": os.path.join(data_dir, "Bob_Weigel_031023_1/GM_CDF")
# }    

def create_2Dplots_ms(info, df, yGSM, zGSM, xmin, xmax, mpline, dtime, normal):
    """2D plots of parameters that change at boundaries, including solar wind
    velocity, pressure, and density.  These plots are used to confirm that the 
    magnetopause is properly identified. The magnetopause (mpline) are plotted on 
    the graphs.  The magnetopause (mpline) is plotted on the graphs, and should be
    where the magnetic pressure is greater than the total pressure (dynamic + 
    thermal pressure).
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       df = dataframe with data to be plotted
       
       yGSM, zGSM = x and y offset for line on which magnetopause locations are 
           found.  That is, we walk along (x, yGSM, zGSM), changing x to find 
           the magnetopause. (GSM)
       
       xmin, xmax = limits of x-axis (GSM)
       
       mpline = x axis coordinate of boundary (magnetosphere) (GSM)
       
       dtime = datetime of associated BATSRUS file
       
       normal = normal to magnetopause at the point that the line ( x[i], y, z ) 
           hits the magnetopause.  np.array of three numbers (GSM)
    
     Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')
    
    # Get angles from x axis to normals to include in titles
    ang = np.rad2deg( np.arccos( np.dot( normal, np.array([1,0,0]) )))
    
    # Create subplots to support multiple graphs on a page
    fig, axs = plt.subplots(6, sharex=True, figsize=(8,10))
    
    
    # Create plots for magnetopause
    df.plot(y=[r'$p_{tot}$', r'$p_{mag}$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax],
                title = 'Magnetopause ' + info['run_name'] + ' ' + str(dtime) + ' [x,' + str(yGSM) +',' 
                    + str(zGSM) + '] ' + "{:.2f}".format(mpline) + r' $R_e$ ' + "{:.2f}".format(ang) 
                    + r'$^{\circ}$', ax=axs[0])
    df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[1])
    df.plot(y=[r'$|B|$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[2])
    df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[3])
    df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[4])
    df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = 'Density $(amu/cc)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[5])
    axs[0].axvline(x = mpline, c='black', ls=':')
    axs[1].axvline(x = mpline, c='black', ls=':')
    axs[2].axvline(x = mpline, c='black', ls=':')
    axs[3].axvline(x = mpline, c='black', ls=':')
    axs[4].axvline(x = mpline, c='black', ls=':')
    axs[5].axvline(x = mpline, c='black', ls=':')
    pltname = 'ms-pp-' + str(dtime) + '-[x,' + str(yGSM) +',' + str(zGSM) + ']' + str([xmin, xmax]) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
    
    plt.close('all')
    return

def create_2Dplots_bs(info, df, yGSM, zGSM, xmin, xmax, bsline, dtime, normal, iteration, angle):
    """2D plots of parameters that change at boundaries, including solar wind
    velocity, pressure, and density.  These plots are used to confirm that the 
    bow shock is properly identified. The bow shock (bsline) is plotted on the 
    graphs.  The bow shock should occur when the solar wind speed perpendicular 
    (u_perp) to the bow shock boundary goes below the magnetosonic speed (Cms)
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       df = dataframe with data to be plotted
       
       yGSM, zGSM = x and y offset for line on which bow shock and magnetopause
           locations are found.  That is, we walk along (x, yGSM, zGSM), changing
           x to find the bow shock and magnetopaus. (GSM)
       
       xmin, xmax = limits of x-axis (GSM)
       
       bsline = x axis coordinate of boundary (bow shock) (GSM)
       
       dtime = datetime of associated BATSRUS file
       
       normalmp = normal to magnetopause at the point that the line ( x[i], y, z ) 
           hits the magnetopause.  np.array of three numbers (GSM)
        
       normal = normal to bow shock at the point that the line ( x[i], y, z ) 
            hits the bow shock.  np.array of three numbers (GSM)

    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')
    
    # Get angles from x axis to normals to include in titles
    ang = np.rad2deg( np.arccos( np.dot( normal, np.array([1,0,0]) )))
    
    # Create subplots to support multiple graphs on a page
    fig, axs = plt.subplots(6, sharex=True, figsize=(8,10))
    
    # Create plots for bow shock
    df.plot(y=[r'$u_{bs\perp}$', r'$c_{MS}$'], use_index=True, \
                ylabel = 'Speed $(km/s)$', xlabel = r'$x_{GSM}$ $(R_E)$', xlim = [xmin, xmax],
                title = 'Bow Shock ' + info['run_name'] + ' ' + str(dtime) + ' [x,' + str(yGSM) +',' + str(zGSM) + '] ' 
                    + "{:.2f}".format(bsline) + r' $R_e$ ' + "{:.2f}".format(ang) + r'$^{\circ}$' + 'Iter: ' + str(iteration),\
                    ax=axs[0])
    df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[1])
    df.plot(y=[r'$|B|$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[2])
    df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[3])
    df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[4])
    df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = 'Density $(amu/cc)$', xlabel = '$x_{GSM} (Re)$', xlim = [xmin, xmax], ax=axs[5])
    axs[0].axvline(x = bsline, c='black', ls=':')
    axs[1].axvline(x = bsline, c='black', ls=':')
    axs[2].axvline(x = bsline, c='black', ls=':')
    axs[3].axvline(x = bsline, c='black', ls=':')
    axs[4].axvline(x = bsline, c='black', ls=':')
    axs[5].axvline(x = bsline, c='black', ls=':')
    pltname = 'ms-u-' + str(dtime) + '-[x,' + str(yGSM) + ','  + str(zGSM) + ']'+ str([xmin, xmax]) \
        + 'Iter-' + str(iteration) + '-' + str(angle) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
            
    plt.close('all')
    return

def create_2Dplots_ns(info, df, xGSM, yGSM, zmin, zmax, nsline, dtime):
    """2D plots of parameters that change at boundaries, including solar wind
    velocity, pressure, and density.  These plots are used to confirm that the
    neutral sheet is located. The neutral sheet (nsline) is plotted on the graphs.  
    The neutral sheet should occur when the magnetic field switches directions, 
    i.e., Bx = 0
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       df = dataframe with data to be plotted
       
       xGSM, yGSM = x and y offset for line on which neutral sheet locations 
           are found.  That is, we walk along (xGSM, yGSM, z), changing
           z to find the neutral sheet. (GSM)
       
       zmin, zmax = limits of z-axis (GSM)
       
       nsline = z axis coordinate of boundary (neutral sheet) (GSM)
       
       dtime = datetime of associated BATSRUS file
       
    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')
    
    # Create subplots to support multiple graphs on a page
    fig, axs = plt.subplots(6, sharex=True, figsize=(8,10))
    
    # Create plots for bow shock
    df.plot(y=[r'$B_x$'], use_index=True, \
                ylabel = '$B_x (nT)$', xlabel = r'$z_{GSM}$ $(R_E)$', xlim = [zmin, zmax],
                title = 'Neutral Sheet ' + info['run_name'] + ' ' + str(dtime) + ' [' + str(xGSM)+ ',' + str(yGSM) +',z] ' 
                    + "{:.2f}".format(nsline) + r' $R_e$ ', ax=axs[0])
    df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = 'z_{GSM} (Re)', xlim = [zmin, zmax], ax=axs[1])
    df.plot(y=[r'$|B|$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'z_{GSM} (Re)', xlim = [zmin, zmax], ax=axs[2])
    df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'z_{GSM} (Re)', xlim = [zmin, zmax], ax=axs[3])
    df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = 'z_{GSM} (Re)', xlim = [zmin, zmax], ax=axs[4])
    df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = 'Density $(amu/cc)$', xlabel = 'z_{GSM} (Re)', xlim = [zmin, zmax], ax=axs[5])
    axs[0].axvline(x = nsline, c='black', ls=':')
    axs[1].axvline(x = nsline, c='black', ls=':')
    axs[2].axvline(x = nsline, c='black', ls=':')
    axs[3].axvline(x = nsline, c='black', ls=':')
    axs[4].axvline(x = nsline, c='black', ls=':')
    axs[5].axvline(x = nsline, c='black', ls=':')
    pltname = 'ms-bx-' + str(dtime) + '-[' + str(xGSM) + ',' + str(yGSM) + ',z]'+ str([zmin, zmax]) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
            
    plt.close('all')
    return


def create_wireframe_plots_mp(info, basename, max_yz, num_yz_pts, xmesh, ymesh, zmesh, dtime, ang):
    """Wireframe plots of magnetopause.  Illustrates the 3D boundary.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above

       basename = basename of file that was processed
       
       max_yz = grid ranges from -max_yz to +max_yz along y axis and along z axis (GSM)

       num_yz_pts = number of points in mesh grid = num_yz_pts x num_yz_pts x num_yz_pts
       
       xmesh, ymesh, zmesh = x,y,z coordinates of wireframe of magnetopause (GSM)
              
       dtime = datetime of BATSRUS file, used in titles and filenames
       
       ang = angle (deg) between the x axis and the normal to magnetopause (GSM)
       
    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')

    ymesh2 = ymesh.reshape((num_yz_pts,num_yz_pts))
    zmesh2 = zmesh.reshape((num_yz_pts,num_yz_pts))
    xmesh2 = xmesh.reshape((num_yz_pts,num_yz_pts))
    ang2 = ang.reshape((num_yz_pts,num_yz_pts))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmesh2, ymesh2, zmesh2, label='Magnetopause')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Magnetopause' )
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '.magnetopause.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(ang2, ymesh2, zmesh2, label='Magnetopause')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Magnetopause' )
    ax.set_xlabel( r'$acos( \hat n \cdot \hat x ) (Deg)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname =  basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '.angle.magnetopause.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
   
    # plt.close('all')
    return

def create_wireframe_plots_bs(info, basename, max_yz, num_yz_pts, xmesh, ymesh, 
                              zmesh, dtime, ang, angle):
    """Wireframe plots of bow shock.  Illustrate the 3D boundaries.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
       
       basename = basename of file that was processed
       
       max_yz = grid ranges from -max_yz to +max_yz along y axis and along z axis (GSM)

       num_yz_pts = number of points in mesh grid = num_yz_pts x num_yz_pts x num_yz_pts
       
       xmesh, ymesh, zmesh = x,y,z coordinates of wireframe for bow shock (GSM)
              
       dtime = datetime of BATSRUS file, used in titles and filenames
       
       ang = angle (deg) between the x axis and the normal to bow shock (GSM)
       
       angle = angle (degrees) that bow shock makes with x axis.  Positive
           angle determined by righthand rule about positive y-axis.  If None,
           angle ignored

    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')

    ymesh2 = ymesh.reshape((num_yz_pts,num_yz_pts))
    zmesh2 = zmesh.reshape((num_yz_pts,num_yz_pts))
    xmesh2 = xmesh.reshape((num_yz_pts,num_yz_pts))
    ang2 = ang.reshape((num_yz_pts,num_yz_pts))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmesh2, ymesh2, zmesh2, label='Bow Shock')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Bow Shock' )
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.bowshock.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(ang2, ymesh2, zmesh2, label='Bow Shock')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Bow Shock' )
    ax.set_xlabel( r'$acos( \hat n \cdot \hat x ) (Deg)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.angle.bowshock.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
    
    # plt.close('all')
    return

def create_wireframe_plots_ns(info, basename, max_xy, num_xy_pts, xmesh, ymesh, zmesh, dtime):
    """Wireframe plots of neutral sheet.  Illustrate the 3D boundaries
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
       
       basename = basename of file that was processed
       
       max_xy = grid ranges -max_xy to 0 along x axis and from -max_xy to 
           +max_xy along y axis (GSM)

        num_yz_pts = number of points in mesh grid = num_yz_pts x num_yz_pts x num_yz_pts
       
       xmesh, ymesh zmesh = x,y,z coordinates of wireframe, the same for neutral 
           sheet (GSM)
                     
       dtime = datetime of BATSRUS file, used in titles and filenames
       
    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp-bs-ns/')

    xmesh2 = xmesh.reshape((num_xy_pts,num_xy_pts))
    ymesh2 = ymesh.reshape((num_xy_pts,num_xy_pts))
    zmesh2 = zmesh.reshape((num_xy_pts,num_xy_pts))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmesh2, ymesh2, zmesh2, label='Neutral Sheet')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Neutral Sheet' )
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = basename + '.' + str(max_xy) + '-' + str(num_xy_pts) + '.neutralsheet.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pltname ) )
        
    # plt.close('all')
    return

def create_boundary_dataframe_mp(batsrus, num_x_points, xmin, xmax, delx, y, z, normal):
    """ Create dataframe based on data from BATSRUS file.  In addition to the 
    data read from the file, the dataframe includes calculated quantities to 
    determine the boundary of the magnetopause.
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
           file
       
       num_x_points = the number of points along the x axis at which data will
           be interpolated (GSM)
       
       xmin, xmax = limits of interpolation along x-axis (GSM)
       
       delx = spacing between points parallel to x-axis (GSM)
       
       y, z = y and z offset.  Points will be interpolated at x[i] along x[i],y,z (GSM)
       
       normal = normal to magnetopause at the point that the line ( x[i], y, z ) 
           hits the magnetopause.  np.array of three numbers (GSM)
               
    Outputs:
        df = the dataframe with BATSRUS and calculated quantaties
    """
    # Set up data storage
    x = np.zeros(num_x_points)

    bx = np.zeros(num_x_points)
    by = np.zeros(num_x_points)
    bz = np.zeros(num_x_points)

    ux = np.zeros(num_x_points)
    uy = np.zeros(num_x_points)
    uz = np.zeros(num_x_points)

    p = np.zeros(num_x_points)

    rho = np.zeros(num_x_points)

    # Loop through the range steps, determining variable values at each
    # point on the x[i],y,z line
    for i in range(num_x_points):  
        x[i] = xmin + delx*i
        XGSM = np.array([x[i], y, z])
        
        bx[i] = batsrus.interpolate( XGSM, 'bx' )
        by[i] = batsrus.interpolate( XGSM, 'by' )
        bz[i] = batsrus.interpolate( XGSM, 'bz' )

        ux[i] = batsrus.interpolate( XGSM, 'ux' )
        uy[i] = batsrus.interpolate( XGSM, 'uy' )
        uz[i] = batsrus.interpolate( XGSM, 'uz' )

        p[i] = batsrus.interpolate( XGSM, 'p' )
                            
        rho[i] = batsrus.interpolate( XGSM, 'rho' )
                            
    # Store variables in dataframe, and calculate some quantities
    df = pd.DataFrame()
    
    df[r'x'] = x
    
    df[r'$B_x$'] = bx
    df[r'$B_y$'] = by
    df[r'$B_z$'] = bz

    df[r'$u_x$'] = ux
    df[r'$u_y$'] = uy
    df[r'$u_z$'] = uz

    df[r'$p$'] = p
    
    df[r'$\rho$'] = rho

    # In MHD papers, when they talk about dynamic pressure, they're
    # generally talking about ram pressure, also known as momentum flux
    #
    # Ram pressure is pdyn = rho * u^2, whereas dynamic pressure that is 
    # discussed in aerodynamics is q = 1/2 rho * u^2
    #
    # If rho is in amu/cc, then amu/cc => 1.66 * 10^-21 kg/m^3
    # if u in km/s, then ==> 1000 m/s, remember we have u^2 in dynamic pressure
    # if we use nPascals ==> 10^9 nPa
    # 
    # 1.66*10-21 * (1000)^2 * 10^9 = 1.66*10^-6 conversion factor
    #
    # df[r'$p_{dyn}$'] = rho * (df[r'$u_x$']**2 + df[r'$u_y$']**2 + df[r'$u_z$']**2) \
    #     * 1.66 * 10**(-6)
    # 
    # In this case, we focus on the solar wind speed normal to the magnetopause
    # (See Basic Space Plasma Physics by Baumjohann and Treumann, 1997, page 188, eqn 8.80)
    df[r'$p_{dyn}$'] = KAPPA * rho * ((normal[0] * df[r'$u_x$'])**2 + 
                                      (normal[1] * df[r'$u_y$'])**2 + 
                                      (normal[2] * df[r'$u_z$'])**2) * 1.66 * 10**(-6)

    df[r'$p_{tot}$'] = df[r'$p$'] + df[r'$p_{dyn}$']
    
    df[r'$|B|$'] = np.sqrt( bx**2 + by**2 + bz**2 )
    df[r'$|u|$'] = np.sqrt( ux**2 + uy**2 + uz**2 )
    
    # Determine magnetic pressure = |normal x B|^2 / 2 / mu0
    #
    # We look at the tangential component of B because there is no normal component
    # at the magnetopause (see Basic Space Plasma Physics from above).
    #
    # (10**-9)**2 to convert from nT**2 to T**2, then 10**9 to convert to nPa.
    # giving 10**-9 total conversion factor
    #
    # df[r'$p_{mag}$'] = df[r'$|B|$']**2 * 10**-9 / 2 / MU0
    #
    # (See Basic Space Plasma Physics by Baumjohann and Treumann, 1997, page 188, eqn 8.80)
    df[r'$p_{mag}$'] = ((normal[1] * df[r'$B_z$'] - normal[2] * df[r'$B_y$'] )**2 +
                        (normal[2] * df[r'$B_x$'] - normal[0] * df[r'$B_z$'] )**2 +
                        (normal[0] * df[r'$B_y$'] - normal[1] * df[r'$B_x$'] )**2) \
                        * 10**-9 / 2 / MU0
    
    # The magnetopause is where magnetic pressure equals dynamic plus thermal 
    # pressure (total pressure).
    # df[r'$p_{dyn} - p_{mag}$'] = df[r'$p_{dyn}$'] - df[r'$p_{mag}$']
    df[r'$p_{tot} - p_{mag}$'] = df[r'$p_{tot}$'] - df[r'$p_{mag}$']
        
    return df

def create_boundary_dataframe_bs(batsrus, num_x_points, xmin, xmax, delx, y, z, 
                                 normal, angle=None):
    """Create dataframe based on data from BATSRUS file.  In addition to the 
    data read from the file, the dataframe includes calculated quantities to 
    determine the boundary of the bow shock.
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
           file
       
       num_x_points = the number of points along the x axis at which data will
           be interpolated (GSM)
       
       xmin, xmax = limits of interpolation along x-axis (GSM)
       
       delx = spacing between points parallel to x-axis (GSM)
       
       y, z = y and z offset.  Points will be interpolated at x[i] along x[i],y,z (GSM)
       
       normal = normal to bow shock at the point that the line ( x[i], y, z ) 
           hits the bow shock.  np.array of three numbers (GSM)
           
       angle = angle (degrees) that bow shock makes with x axis.  Positive
           angle determined by righthand rule about positive y-axis.  If None,
           angle ignored
               
    Outputs:
        df = the dataframe with BATSRUS and calculated quantaties
    """
    # Set up data storage
    x = np.zeros(num_x_points)

    bx = np.zeros(num_x_points)
    by = np.zeros(num_x_points)
    bz = np.zeros(num_x_points)

    ux = np.zeros(num_x_points)
    uy = np.zeros(num_x_points)
    uz = np.zeros(num_x_points)

    p = np.zeros(num_x_points)

    rho = np.zeros(num_x_points)

    # Loop through the range steps, determining variable values at each
    # point on the x[i],y,z line
    for i in range(num_x_points):  
        x[i] = xmin + delx*i
        
        # if angle is None, lines being traced are parallel to the x axis
        # otherwise, they are at an angle and we need to adjust x and z
        if angle is None:
            XGSM = np.array([x[i], y, z])
        else:
            anglerad = np.deg2rad(angle)
            xtmp = x[i]*np.cos(anglerad) + z*np.sin(anglerad)
            ztmp = z*np.cos(anglerad)    - x[i]*np.sin(anglerad)
            XGSM = np.array([xtmp, y, ztmp])
        
        # Make sure we're within the simulation volume
        if (XGSM[0] < batsrus.xGlobalMin or XGSM[0] > batsrus.xGlobalMax) or \
            (XGSM[1] < batsrus.yGlobalMin or XGSM[1] > batsrus.yGlobalMax) or \
            (XGSM[2] < batsrus.zGlobalMin or XGSM[2] > batsrus.zGlobalMax):

            bx[i] = np.nan
            by[i] = np.nan
            bz[i] = np.nan

            ux[i] = np.nan
            uy[i] = np.nan
            uz[i] = np.nan

            p[i] = np.nan
                                
            rho[i] = np.nan
        else:
            bx[i] = batsrus.interpolate( XGSM, 'bx' )
            by[i] = batsrus.interpolate( XGSM, 'by' )
            bz[i] = batsrus.interpolate( XGSM, 'bz' )
    
            ux[i] = batsrus.interpolate( XGSM, 'ux' )
            uy[i] = batsrus.interpolate( XGSM, 'uy' )
            uz[i] = batsrus.interpolate( XGSM, 'uz' )
    
            p[i] = batsrus.interpolate( XGSM, 'p' )
                                
            rho[i] = batsrus.interpolate( XGSM, 'rho' )
                            
    # Store variables in dataframe, and calculate some quantities
    df = pd.DataFrame()
    
    df[r'x'] = x
    
    df[r'$B_x$'] = bx
    df[r'$B_y$'] = by
    df[r'$B_z$'] = bz

    df[r'$u_x$'] = ux
    df[r'$u_y$'] = uy
    df[r'$u_z$'] = uz

    df[r'$p$'] = p
    
    df[r'$\rho$'] = rho

    df[r'$|B|$'] = np.sqrt( bx**2 + by**2 + bz**2 )
    df[r'$|u|$'] = np.sqrt( ux**2 + uy**2 + uz**2 )
    
    # c_s = speed of sound in km/sec
    #
    # p in nPa => 10^-9 Pa
    # rho in amu/cc,  amu/cc => 1.66 * 10^-21 kg/m^3
    # which gives m^2/s^2, we want km^2/s^2, so a factor of 10^-6
    #
    # 10^-6 * 10^-9 / 1.66 x 10^-21 = 6.024 x 10^5
    df[r'$c_s^2$'] = GAMMA * df[r'$p$'] / df[r'$\rho$'] * 6.024 * 10**5
    
    # v_a = Alfven velocity in km/sec
    # 
    # (10**-9)**2 to convert B^2 from nT**2 to T**2
    # rho in amu/cc,  amu/cc => 1.66 * 10^-21 kg/m^3
    # which gives m^2/s^2, we want km^2/s^2, so a factor of 10^-6
    #
    # conversion factor 10^-6 * 10^-18 / 1.66 x 10^-21 = 6.024 x 10^-4
    df[r'$v_a^2$'] = df[r'$|B|$']**2 * 6.024 * 10**-4 / MU0 / df[r'$\rho$']
    
    # Determine the magnetosonic speed c_{MS}
    df[r'$c_{MS}$'] = np.sqrt( df[r'$c_s^2$'] + df[r'$v_a^2$'] )
    
    # The solar wind speed normal to the bow shock
    # (See Basic Space Plasma Physics by Baumjohann and Treumann, 1997, page 182)
    df[r'$u_{bs\perp}$'] = np.abs( normal[0] * df[r'$u_x$'] +
                             normal[1] * df[r'$u_y$'] +
                             normal[2] * df[r'$u_z$'] )
    
    # Determine difference between magnetosonic speed and solar wind speed.
    # We will use this to find the bow shock.  The bow shock is when the solar
    # wind becomes sub-magnetosonic
    df[r'$c_{MS} - u_{bs\perp}$'] = df[r'$c_{MS}$'] - df[r'$u_{bs\perp}$']
    
    return df

def create_boundary_dataframe_ns(batsrus, num_z_points, zmin, zmax, delz, x, y):
    """ Create dataframe based on data from BATSRUS file.  In addition to the 
    data read from the file, the dataframe includes calculated quantities to 
    determine the neutral sheet boundary.
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
           file
       
       num_z_points = the number of points along the z axis at which data will
           be interpolated (GSM)
       
       zmin, zxmax = limits of interpolation along x-axis (GSM)
       
       delz = spacing between points parallel to x-axis (GSM)
       
       x, y = x and y offset.  Points will be interpolated at z[i] along x,y,z[i] (GSM)
       
    Outputs:
        df = the dataframe with BATSRUS and calculated quantaties
    """
    # Set up data storage
    z = np.zeros(num_z_points)

    bx = np.zeros(num_z_points)
    by = np.zeros(num_z_points)
    bz = np.zeros(num_z_points)

    ux = np.zeros(num_z_points)
    uy = np.zeros(num_z_points)
    uz = np.zeros(num_z_points)

    p = np.zeros(num_z_points)

    rho = np.zeros(num_z_points)

    # Loop through the range steps, determining variable values at each
    # point on the x,y,z[i] line
    for i in range(num_z_points):  
        z[i] = zmin + delz*i
        XGSM = np.array([x, y, z[i]])
        
        # bx is how we find the neutral sheet.  Bx switches direction, so
        # we look for bx = 0
        bx[i] = batsrus.interpolate( XGSM, 'bx' )
        by[i] = batsrus.interpolate( XGSM, 'by' )
        bz[i] = batsrus.interpolate( XGSM, 'bz' )

        ux[i] = batsrus.interpolate( XGSM, 'ux' )
        uy[i] = batsrus.interpolate( XGSM, 'uy' )
        uz[i] = batsrus.interpolate( XGSM, 'uz' )

        p[i] = batsrus.interpolate( XGSM, 'p' )
                            
        rho[i] = batsrus.interpolate( XGSM, 'rho' )
                            
    # Store variables in dataframe, and calculate some quantities
    df = pd.DataFrame()
    
    df[r'z'] = z
    
    df[r'$B_x$'] = bx
    df[r'$B_y$'] = by
    df[r'$B_z$'] = bz

    df[r'$u_x$'] = ux
    df[r'$u_y$'] = uy
    df[r'$u_z$'] = uz

    df[r'$p$'] = p
    
    df[r'$\rho$'] = rho

    df[r'$|B|$'] = np.sqrt( bx**2 + by**2 + bz**2 )
    df[r'$|u|$'] = np.sqrt( ux**2 + uy**2 + uz**2 )
        
    return df

def initialize_xmesh_mp( batsrus, num_x_points, xmin, xmax, delx, ymesh, zmesh):
    """Initialize the x mesh grids used in the interpolations to find the shape
    of the magnetopause. Based on various text books and papers, we assume
    that the magnetopause is a parabola initially.  
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
            file
            
       num_x_pts = the number of points in grid along x axis (GSM)
         
       xmin, xmax = range of x values that we examine in the interpolation, will
           set the limits of the x mesh grid
       
       delx = spacing between points parallel to x-axis (GSM)
       
       ymesh, zmesh = y and z mesh grids. We fill in at the x mesh grid at each
           combination of y,z points (GSM)
      
    Outputs:
        xmesh = x mesh grid for magnetopause (GSM)
        
        xmp = subsolar distance to magnetopause (GSM)
    """

    logging.info('Initializing magnetopause xmesh')

    # Create dataframe for finding the magnetopause.  Dataframe includes data
    # on magnetic pressures, dynamic ram pressure, magnetosonic speed, and
    # solar wind speed used to find boundaries.  We're looking nose on to the
    # boundary, so the normals are (1,0,0) in GSM coordinates
    normal = np.array([1,0,0])
    df = create_boundary_dataframe_mp(batsrus, num_x_points, xmin, xmax, delx, 
                                   0, 0, normal)

    # Walk from xmax toward xmin find the first x value where total pressure 
    # equals magnetic pressure, that is, when p_{tot} - p_{mag} becomes 
    # negative.  This will be the location of the magnetopause
    #
    # df[r'$p_{tot} - p_{mag}$'] = df[r'$p_{tot}$'] - df[r'$p_{mag}$']
    #
    # range(num_x_points-1,-1,-1) handles that ranges are xmin to xmax, and we
    # want to start at xmax
    for q in range(num_x_points-1,-1,-1):
        if df[r'$p_{tot} - p_{mag}$'][q] < 0: 
            xmp = df[r'x'][q]
            break

    xmesh = np.zeros(ymesh.shape)
    
    # Based on the paper, "Magnetopause as conformal mapping," Yasuhito Narita, 
    # Simon Toepfer, and Daniel Schmid, Ann. Geophys., 41, 87–91,
    # In the examples that we used to test this algorithm, alpha tended to be
    # close to 1.  In which case the Shue formala reduces to:
    xmesh = xmp - 1 / 4 / xmp * ( ymesh**2 + zmesh**2 )
    
    return xmesh, xmp

def initialize_xmesh_bs( batsrus, num_x_points, xmin, xmax, delx, ymesh, zmesh):
    """Initialize the x mesh grids used in the interpolations to find the shape
    of bow shock. Based on various text books and papers, we assume
    that the bow shock is a parabola. With the "bottom" of the parabola 
    at (x, 0, 0), and the width at the earth (0, +/-2 x, 0) and (0, 0, +/-2 x) 
    along the y and z axes respectively.  
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
            file
            
       num_x_pts = the number of points in grid along x axis (GSM)
         
       xmin, xmax = range of x values that we examine in the interpolation, will
           set the limits of the x mesh grid
       
       delx = spacing between points parallel to x-axis (GSM)
       
       ymesh, zmesh = y and z mesh grids. We fill in at the x mesh grid at each
           combination of y,z points (GSM)
      
    Outputs:
        xmesh = x mesh grid for bow shock (GSM)
        
        xbs = subsolar distance to bow shock (GSM)
    """

    logging.info('Initializing bow shock xmesh')

    # Create dataframe for finding the bow shock.  Dataframe includes data
    # on magnetic pressures, dynamic ram pressure, magnetosonic speed, and
    # solar wind speed used to find boundaries.  We're looking nose on to the
    # boundaries, so the normals are (1,0,0) in GSM coordinates
    normal = np.array([1,0,0])
    df = create_boundary_dataframe_bs(batsrus, num_x_points, xmin, xmax, delx, 
                                   0, 0, normal)

    # Walk from xmax to xmin and find the first x value where magnetosonic 
    # speed is equal to solar wind speed, that is, when c_{MS} - u_{bs\perp}  
    # becomes positive.  This will be the approximate location of the bow shock
    #
    # df[r'$c_{MS} - u$'] = df[r'$c_{MS}$'] - df[r'$u_{bs\perp}$']
    #
    # range(num_x_points-1,-1,-1) handles that ranges are xmin to xmax, and we
    # want to start at xmax
    for q in range(num_x_points-1,-1,-1):
        if df[r'$c_{MS} - u_{bs\perp}$'][q] > 0: 
            xbs = df[r'x'][q]
            break

    xmesh = np.zeros(ymesh.shape)
    
    # Based on the paper, "Orientation and shape of the Earth's bow shock in three dimensions,"
    # V. Formisano, Planetary and Space Science, Volume 27, Issue 9, September 1979, 
    # Pages 1151-1161, assume that the bow shock is a parabola. the width at 
    # the earth (0, +/-2 x, 0) and (0, 0, +/-2 x) along the y and z axes respectively 
    xmesh = xbs - 1 / 4 / xbs * ( ymesh**2 + zmesh**2 )
    
    return xmesh, xbs

def findboundary_mp(info, filepath, time, max_yz, num_yz_pts, xlimits, num_x_points, maxits, tol, 
                    alpha = 0.7, plotit=False):
    """Find the boundary of the magnetopause. To find the boundary, we iteratively 
    explore a 3D grid.  The equations for the magnetopause boundary depend
    on the solar wind speed normal to the boundary and the magnetic field B parallel 
    to the boundary.  This requires us to know the shape of the boundary to find normals.  
    So we assume a shape, estimate the boundary. Use new shape to determine normals,
    and reestimate boundary. Repeat iteratively until maxits or tolerance tol is reached.
    
    To find the shape, we march down a series of lines parallel to the x axis.
    Along each line, we find the magnetopause.  See create_boundary_dataframe_mp 
    for the boundary criteria.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       filepath = filepath of BATSRUS file 
       
       time = time for BATSRUS file 
       
       max_yz = grid ranges from -max_yz to +max_yz along y axis and along z axis (GSM)
             
       num_yz_pts = y-z plane split into num_yz_pts * num_yz_pts points to create grid (GSM)
       
       xlimits = grid ranges from xlimits[0] to xlimits[1] (GSM)

       num_x_pts = the number of points in grid along x axis (GSM)
     
       maxits = max number of iterations
       
       tol = tolerence, once std deviation between successive iterations drops below
           tol, the iteration loop ends
           
       alpha = initial value of exponent in Shue equation describing magnetopause
           boundary.  Shue equation used to estimate normal to magnetopause
           
       plotit = Boolean, create plots for each line using create_2Dplots
       
    Outputs:
        None - results saved in pickle files and plots
    """
    
    basename = os.path.basename(filepath)
    logging.info(f'Data for BATSRUS file... {basename}')

    # Read BATSRUS file
    batsrus = swmfio.read_batsrus(filepath)
    assert(batsrus != None)
     
    # Convert times to datetime format for use in plots
    dtime = datetime(*time) 
    
    # Set up grid for finding magnetopause
    
    # Look for the magnetopause boundary between the user-specified xmin and xmax
    xmin = xlimits[0]
    xmax = xlimits[1]
    delx = (xmax - xmin)/num_x_points
    
    # # Find dely and delz for the gradients below
    # delyz = 2 * max_yz / ( num_yz_pts - 1 )

    # Create 3D mesh grid where we will record the magnetopause boundary.
    # y-z grid is uniform.  x values initially assume a specific boundary shape, 
    # see initialize_xmesh_mp.  x values will be updated with estimated magnetopause 
    # position
    y = np.linspace(-max_yz, max_yz, num_yz_pts)
    z = np.linspace(-max_yz, max_yz, num_yz_pts)
    ymesh, zmesh = np.meshgrid(y, z)
    
    ymesh = ymesh.reshape(-1)
    zmesh = zmesh.reshape(-1)
    xmesh, xmp = initialize_xmesh_mp( batsrus, num_x_points, xmin, xmax, delx, ymesh, zmesh)

    # Initialize arrays for statistics on convergence
    stdmp = np.full(maxits, np.nan, dtype=np.float32)
    
    # Do an interative process to determine the shape of magnetopause boundary, 
    # specified by xmeshmp.  We start with an assumed shape, recalculate boundary.
    # Used recalculated boundary to iterate again.  Rinse and repeat.
    for l in range(maxits):
        logging.info(f'Iteration {l}...')
        
        if USE_SHUE:
            # Find the normal for the magnetopause.  The numeric gradient won't work 
            # for the magnetopause, it is unstable.  So we smooth the curve by using a 
            # least squares fit of the Shue equation to the data from the previous iteration
            # For the first iteration, l = 0, we used the assumed values from 
            # initialize_xmesh_mp and the input value for alpha
            if l > 0:
                rn = np.sqrt( xmesh**2 + ymesh**2 + zmesh**2 )
                bn = np.log( 2 / (1 + xmesh/rn) )
                b2n = bn * bn
                rrn = np.log( rn / xmp )
                alpha = np.nansum( bn * rrn ) / np.nansum( b2n )
            logging.info(f'alpha: {alpha}')
        else:
            # Find the numerical gradient for the magnetopause, so we can get the 
            # normal to the magnetopause
            delyz = 2 * max_yz / ( num_yz_pts - 1 )
            xmesh2 = xmesh.reshape((num_yz_pts,num_yz_pts))
            gradmp = np.gradient(xmesh2, delyz)
            # d/dy term
            gradmpy = gradmp[1].reshape(-1)
            # d/dz term
            gradmpz = gradmp[0].reshape(-1)

        # Keep a copy so we can see how quickly the iterations converge
        # Nothing to save on l == 0 iteration
        xmesh_old = deepcopy(xmesh)
        
        # Create new storage for mesh.  NaNs specify that the magnetopause was not found.
        xmesh = np.full(ymesh.shape, np.nan, dtype=np.float32)
                
        # Record the angle between the normal and the x axis (1,0,0)
        # We use this to analyze the solution.  At high normal angles,  the magnetopause
        # are oblique to the solar wind.  The dynamic ram pressure depends on 
        # the normal component of the solar wind.  When the magnetopause normal
        # is obliqueu to the solar wind, the dynamic pressure is never high 
        # enough to exceed the magnetic pressure.  
        ang = np.zeros(ymesh.shape)
        
        # Loop thru the lines parallel to x axis that we travel down
        # The y and z offsets from the x axis are stored in ymesh and zmesh
        for m in range(len(ymesh)):
            
            # To find the magnetopause boundary, we need the normal to the boundary
            
            if USE_SHUE:
                # Assume the magnetopause has a shape described by the Shue equation
                # See (Shue et al) 1997 New Functional Form Magnetopause Size & Shape
                # 
                # Use the gradient of that equation to determine the normal.
                rr = np.sqrt( xmesh_old[m]**2 + ymesh[m]**2 + zmesh[m]**2 ) 
                term = 2**alpha * xmp * alpha /(1 + xmesh_old[m]/rr)**(alpha+1)
                normalx = xmesh_old[m]/rr + term*(1-xmesh_old[m]**2/rr**2)/rr
                normaly = ymesh[m]/rr - term*xmesh_old[m]*ymesh[m]/rr**3
                normalz = zmesh[m]/rr - term*xmesh_old[m]*zmesh[m]/rr**3
                normal = np.array([normalx, normaly, normalz]) \
                    / np.sqrt( normalx**2 + normaly**2 + normalz**2 )
            else:         
                # Use gradient to determine normal
                #
                # If the gradient is NaN, we assume that we're on the flanks of the 
                # magnetopause, where the boundary becomes parallel to the solar wind. 
                # So we assume the normal to the boundary has an angle of 85 degrees 
                # from the x axis.  And ensure that the normal is a unit [m].
                if( np.isnan( gradmpy[m] ) or np.isnan( gradmpz[m] ) ):
                    rr = np.sqrt( ymesh[m]**2 + zmesh[m]**2 )
                    xx = rr * np.tan( 0.0875 ) # 5 degrees 
                    normal = np.array( [xx, ymesh[m], zmesh[m] ] ) / np.sqrt( xx**2 + rr**2 )
                else:
                    normal = np.array( [1, -gradmpy[m], -gradmpz[m]] ) \
                        / np.sqrt( 1 + gradmpy[m]**2 + gradmpz[m]**2 )
            
            # Record the angles between the normals and the x axis (1,0,0) 
            # We use this in some plots.
            ang[m] = np.rad2deg( np.arccos( np.dot( normal, np.array([1,0,0]) )))

            # Create dataframe for finding the boundary.  Dataframe includes data
            # on magnetic pressure, dynamic ram pressure, etc. used to find magnetopause
            df = create_boundary_dataframe_mp(batsrus, num_x_points, xmin, xmax, delx, 
                                           ymesh[m], zmesh[m], normal)
            
            # Walk from xmax toward xmin and find the first x value where total 
            # pressure (dynamic ram + thermal pressure) equals magnetic pressure, 
            # that is, when p_{tot} - p_{mag} becomes negative.  This will be 
            # the approximate location of the magnetopause
            #
            # df[r'$p_{tot} - p_{mag}$'] = df[r'$p_{tot}$'] - df[r'$p_{mag}$']
            #
            # range(num_x_points-1,-1,-1) handles that ranges are xmin to xmax, and we
            # want to start at xmax
            #
            # Only do loop if $p_{tot} starts out larger than p_{mag}.  If it
            # is smaller at the start, we'll never have a transition to a value
            # smaller than p_{mag}
            # if df[r'$p_{dyn} - p_{mag}$'][num_x_points-1] > 0:
            if df[r'$p_{tot} - p_{mag}$'][num_x_points-1] > 0:
                for q in range(num_x_points-1,-1,-1):
                    if df[r'$p_{tot} - p_{mag}$'][q] <= 0: 
                        xmesh[m] = df[r'x'][q]
                        break
                
            df.set_index(r'x', inplace=True)

            # Create plots that we visually inspect to determine the 
            # magnetopause boundary selection
            if plotit:
                create_2Dplots_ms(info, df, ymesh[m], zmesh[m], xmin, xmax, xmesh[m], 
                              dtime, normal )

        logging.info(f'Iteration: {l} Std Dev MP Diff: {np.nanstd(xmesh - xmesh_old)}')
        
        # Calculate statistics on convergence
        # Note, nothing to calculate for l == 0 iteration
        stdmp[l] = np.nanstd(xmesh - xmesh_old)

        # If both the magnetosphere are known within tolerance (tol),
        # exit loop
        if np.nanstd(xmesh - xmesh_old) < tol:
            break
 
    # Create 3D wireframe plots for the magnetopause and the bow shock
    create_wireframe_plots_mp(info, basename, max_yz, num_yz_pts, xmesh, ymesh, zmesh, dtime, ang)
    
    # Save magnetopause results to pickle files
    create_directory(info['dir_derived'], 'mp-bs-ns')
    pklname = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '.magnetopause.pkl'
    
    dfmp = pd.DataFrame()
    dfmp['y'] = ymesh
    dfmp['z'] = zmesh
    dfmp['x'] = xmesh
    dfmp['angle'] = ang
    dfmp['abs diff'] = np.abs(xmesh - xmesh_old)

    dfmp.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', pklname) )

    pklname2 = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '.stats.magnetopause.pkl'
    pklname3 = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '.stats.magnetopause.png'
 
    dfstats = pd.DataFrame()
    dfstats['STD MP'] = stdmp

    dfstats.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', pklname2) )
    
    # Also store the data in a VTK file
    xyz = ['x', 'y', 'z']
    colorvars = ['angle', 'abs diff']
   
    vtkname = basename + '-' + str(max_yz) + '-' + str(num_yz_pts) + '.magnetopause'
   
    vtk_mp = sqwireframe( dfmp, xyz, colorvars, num_yz_pts )
    vtk_mp.convert_to_vtk()
    vtk_mp.write_vtk_to_file( os.path.join( info['dir_derived'], 'mp-bs-ns' ), vtkname, 'wireframe' )
 
    # Save data for Shue shape
    if USE_SHUE:
        # Update Shue fit, get new alpha
        rn = np.sqrt( xmesh**2 + ymesh**2 + zmesh**2 )
        bn = np.log( 2 / (1 + xmesh/rn) )
        b2n = bn * bn
        rrn = np.log( rn / xmp )
        alpha = np.nansum( bn * rrn ) / np.nansum( b2n )
        logging.info(f'Final alpha: {alpha}')

        # Use scipy fsolve to solve Shue eqn for x, we'll use these x's to
        # create a meshgrid of the Shue fit
        from scipy.optimize import fsolve
        
        # Need this function for fsolve.  It's the Shue eqn
        def shuefunc( x0, *params ):
            alpha0, y0, z0, rmp0 = params
            r0 = np.sqrt( x0**2 + y0**2 + z0**2 )
            return r0 - rmp0 * ( 2/(1 + x0/r0) )**alpha0
            
        # Create mesh of Shue fit
        xmeshshue = np.zeros( ymesh.shape ) 
        for m in range( len(xmeshshue) ):
            params = ( alpha, ymesh[m], zmesh[m], xmp )
            # Use xmesh as start point 
            xmeshshue[m] = fsolve( shuefunc, xmesh[m], args=params, xtol=tol )

        dfmpshue = pd.DataFrame()
        dfmpshue['y'] = ymesh
        dfmpshue['z'] = zmesh
        dfmpshue['x'] = xmeshshue
        dfmpshue['r'] = np.sqrt( xmeshshue**2 + ymesh**2 + zmesh**2 )
        # angle x,y,z radius makes to x-sxis, arccos of x,y,z dot product with [1,0,0]
        # This angle is NOT the normal
        dfmpshue['angle'] = np.rad2deg( np.arccos( dfmpshue['x']/dfmpshue['r'] ) ) 
        
        xyz = ['x', 'y', 'z']
        colorvars = ['angle']
       
        vtknameshue = basename + '-' + str(max_yz) + '-' + str(num_yz_pts) + '.magnetopause-shue'
       
        vtk_mp = sqwireframe( dfmpshue, xyz, colorvars, num_yz_pts )
        vtk_mp.convert_to_vtk()
        vtk_mp.write_vtk_to_file( os.path.join( info['dir_derived'], 'mp-bs-ns' ), vtknameshue, 'wireframe' )

    # Create plot of convergence stats
    fig, ax = plt.subplots()   
    dfstats.plot(y=['STD MP'], title='Std Dev of Diff. Successive Iterations (Magnetopause)', 
              ylabel = 'Std Deviation', xlabel= 'Iteration',ax=ax)
    ax.axhline( y = tol, ls =":")
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pklname3 ) )

    return dfmp

def findboundary_bs(info, filepath, time, max_yz, num_yz_pts, xlimits, num_x_points, 
                    maxits, tol, plotit=False, angle=None):
    """Find the boundary of the bow shock. To find the boundary, we iteratively 
    explore a 3D grid.  The equations for the bow shock on the solar wind speed 
    normal to the boundary. This requires us to know the shape of the boundary
    to find normals.  So we assume a shape, estimate the boundary. Use the new
    shape to determine normals, and reestimate boundary.  Repeat iteratively 
    until maxits or tolerance tol is reached.
    
    To find the shape, we march down a series of lines parallel to the x axis.
    Along each line, we find where the bow shock is located.  See
    create_boundary_dataframe_bs for the boundary criteria.
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       filepath = filepath of BATSRUS file 
       
       time = time for BATSRUS file 
       
       max_yz = grid ranges from -max_yz to +max_yz along y axis and along z axis (GSM)
             
       num_yz_pts = y-z plane split into num_yz_pts * num_yz_pts points to create grid (GSM)
       
       xlimits = grid ranges from xlimits[0] to xlimits[1] (GSM)

       num_x_pts = the number of points in grid along x axis (GSM)
     
       maxits = max number of iterations
       
       tol = tolerence, once std deviation between successive iterations drops below
           tol, the iteration loop ends
           
       plotit = Boolean, create plots for each line using create_2Dplots
       
       angle = angle (degrees) that bow shock makes with x axis.  Positive
           angle determined by righthand rule about positive y-axis.  If None,
           angle ignored
           
    Outputs:
        None - results saved in pickle files and plots
    """
    
    # angle only implemented for USE_GEN_PARAMETER
    if not angle is None:
        assert( USE_GEN_PARAMETER )
    
    basename = os.path.basename(filepath)
    logging.info(f'Data for BATSRUS file... {basename}')

    # Read BATSRUS file
    batsrus = swmfio.read_batsrus(filepath)
    assert(batsrus != None)
     
    # Convert times to datetime format for use in plots
    dtime = datetime(*time) 
    
    # Set up grid for finding bow shock
    
    # Look for the bow shock boundary between the user-specified xmin and xmax
    xmin = xlimits[0]
    xmax = xlimits[1]
    delx = (xmax - xmin)/num_x_points
    
    # Create 3D mesh grid where we will record the bow shock boundary.
    # y-z grid is uniform.  x values initially assume a specific boundary shape, 
    # see initialize_xmesh_bs.  x values will be updated with estimated bow shock
    # boundary
    y = np.linspace(-max_yz, max_yz, num_yz_pts)
    z = np.linspace(-max_yz, max_yz, num_yz_pts)
    ymesh, zmesh = np.meshgrid(y, z)
    
    ymesh = ymesh.reshape(-1)
    zmesh = zmesh.reshape(-1)
    # Note, we ignore angle in initialization, assume bow shock is parallel to x axis
    xmesh, xbs = initialize_xmesh_bs( batsrus, num_x_points, xmin, xmax, delx, 
                                     ymesh, zmesh)

    # Initialize array for statistics
    stdbs = np.full(maxits, np.nan, dtype=np.float32)
    
    # We will need angle in radians below
    if not angle is None:
        anglerad = np.deg2rad(angle)

    # Do an interative process to determine the shape of bow shock boundary, 
    # specified by xmeshbs.  We start with an assumed shape, recalculate boundary.
    # Used recalculated boundary to iterate again.  Rinse and repeat.
    for l in range(maxits):
        logging.info(f'Iteration {l}...')

        # Find the normal for the bow shock.  The numerical gradient won't work 
        # for the bow shock, it is unstable.  So we smooth the curve by using a 
        # least squares fit of a parabola of revolution to the data from the
        # previous iteration
        if USE_GEN_PARAMETER:
            # Use general quadratic conic section fit for bow shock (no x^2 term)
            #
            # a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z + k = 0 
            #
            # We divide by k and redefine coeficients to get
            #
            # => a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z - 1 = 0
            #
            xn = deepcopy( xmesh.reshape(-1) )
            yn = deepcopy( ymesh )
            zn = deepcopy( zmesh )
            
            # in fit, ignore entries where we have xn = NaN
            yn = np.delete( yn, np.argwhere( np.isnan(xn) ) ) 
            zn = np.delete( zn, np.argwhere( np.isnan(xn) ) ) 
            xn = np.delete( xn, np.argwhere( np.isnan(xn) ) ) 
            
            # if angle is not None, the lines used to find the bow shock are
            # tilted with respect to x axis.  If angle is None, they are parallel 
            # to the x axis and nothing needs to be done.
            if not angle is None:
                xtmp = deepcopy( xn )
                ztmp = deepcopy( zn )
                xn = xtmp*np.cos(anglerad) + ztmp*np.sin(anglerad)
                zn = ztmp*np.cos(anglerad) - xtmp*np.sin(anglerad)
                
            # Use matrix math to do least squares fit to data. AA coeff = 1, 
            # which is the quadratic eqn above in matrix form
            k = np.ones(len(xn))
            AA = np.vstack([xn**2, yn**2, zn**2, xn*yn, xn*zn, yn*zn, xn, yn, zn]).T 
            coeff = np.linalg.lstsq( AA, k, rcond=None )[0] # We only need the coefficients
            a0, b0,c0,d0,e0,f0,g0,h0,i0 = coeff
            logging.info(f'a: {a0}, b: {b0}, c: {c0}, d: {d0}, e: {e0}, f: {f0}, g: {g0}, h: {h0}, i: {i0}')

        if USE_TWO_PARAMETER:
            # Two parameter parabola
            #
            # xn = xbs + A yn^2 + B zn^2 
            # where A = (Sum_n zn^4 Sum_n (xn - xbs)*yn^2 - Sum_n yn^2*zn^2 Sum_n (xn - xbs)*zn^2)
            #           / ( Sum_n yn^4 Sum_n zn^4 - (Sum_n yn^2*zn^2)^2 )
            #
            #       B = (Sum_n yn^4 Sum_n (xn - xbs)*zn^2 - Sum_n yn^2*zn^2 Sum_n (xn - xbs)*yn^2)
            #           / ( Sum_n yn^4 Sum_n zn^4 - (Sum_n yn^2*zn^2)^2 )
            xn = xmesh.reshape(-1) - xbs
            yn = deepcopy( ymesh )
            zn = deepcopy( zmesh )
            yn[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN
            zn[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN

            y2n = yn**2
            z2n = zn**2
            y4n = yn**4
            z4n = zn**4
            xy2n = xn*y2n
            xz2n = xn*z2n
            y2z2n = y2n*z2n
            
            y4ns = np.nansum(y4n)
            z4ns = np.nansum(z4n)
            xy2ns = np.nansum(xy2n)
            xz2ns = np.nansum(xz2n)
            y2z2ns = np.nansum(y2z2n)
            
            denom = y4ns * z4ns - y2z2ns**2
            
            A = (z4ns * xy2ns - y2z2ns * xz2ns)/denom
            B = (y4ns * xz2ns - y2z2ns * xy2ns)/denom
            logging.info(f'A: {A}, B: {B}')
            
        if USE_ONE_PARAMETER:
            # One parameter parabola
            #
            # xn = xbs + A (yn^2 + zn^2) 
            # where A = Sum_n (xn - xbs)(yn^2 + zn^2) / Sum_n (yn^2 + zn^2)^2
            #
            xn = xmesh.reshape(-1) - xbs
            r2n = ymesh**2 + zmesh**2
            r2n[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN
            A = np.nansum( xn * r2n ) / np.nansum( r2n**2 )
            logging.info(f'A: {A}')
        
        # Keep a copy so we can see how quickly the iterations converge
        xmesh_old = deepcopy(xmesh)
        
        # Create new storage for mesh.  NaNs specify that the bow shock
        # boundary was not found along a given y-z line
        xmesh = np.full(ymesh.shape, np.nan, dtype=np.float32)
                
        # Record the angle between the normal and the x axis (1,0,0)
        # We use this to analyze the solution. We know that the bow shock
        # breaks down around 80 deg.  At that angle the bow shock is oblique to 
        # the solar wind.  Thus, the solar wind normal to the bow shock is so 
        # small that it is never super-magnetosonic, and hence
        # can't go sub-magnetosonic.  
        ang = np.zeros(ymesh.shape)
        
        if USE_GEN_PARAMETER:
            # Need this function for fsolve.  It's the general quadratic eqn
            # It's used to find points on the quadratic surface
            def genfunc( x0, params ):
                a,b,c,d,e,f,g,h,i,y0,z0,angle = params
                if angle is None:
                    return a*x0**2 + b*y0**2 +c*z0**2 + d*x0*y0 + e*x0*z0 + \
                            f*y0*z0 + g*x0 + h*y0 + i*z0 - 1
                else:
                    # Rotate about the y-axis
                    angrad = np.deg2rad(angle)
                    x1 = x0*np.cos(angrad) + z0*np.sin(angrad)
                    z1 = z0*np.cos(angrad) - x0*np.sin(angrad)
                    return a*x1**2 + b*y0**2 +c*z1**2 + d*x1*y0 + e*x1*z1 + \
                            f*y0*z1 + g*x1 + h*y0 + i*z1 - 1
           
        # Loop thru the lines parallel to x axis that we travel down
        # The y and z offsets from the x axis are stored in ymesh and zmesh
        for m in range(len(ymesh)):
            
            # To find the bow shock boundary, we need the normal to the boundary
            # Normal is gradient of assumed parabolic shape, see above
            if USE_GEN_PARAMETER:
                # Use least squares fit to calc normal
                params = coeff.tolist()
                params.append(ymesh[m])
                params.append(zmesh[m])
                params.append(angle)

                if np.isnan( xmesh[m] ): 
                    xguess = -50. # If its NaN, assume an initial value
                else: 
                    xguess = xmesh[m]
                
                xtmp = fsolve( genfunc, xguess, args=params, xtol=tol )[0]

                if angle is None:
                    normal = np.array( [2*coeff[0]*xtmp     + coeff[3]*ymesh[m] + coeff[4]*zmesh[m] + coeff[6],
                                        2*coeff[1]*ymesh[m] + coeff[3]*xtmp     + coeff[5]*zmesh[m] + coeff[7],
                                        2*coeff[2]*zmesh[m] + coeff[4]*xtmp     + coeff[5]*ymesh[m] + coeff[8]] )
                else:
                    xang = xtmp*np.cos(anglerad)     + zmesh[m]*np.sin(anglerad) 
                    zang = zmesh[m]*np.cos(anglerad) - xtmp*np.sin(anglerad) 
                    normal = np.array( [2*coeff[0]*xang     + coeff[3]*ymesh[m] + coeff[4]*zang     + coeff[6],
                                        2*coeff[1]*ymesh[m] + coeff[3]*xang     + coeff[5]*zang     + coeff[7],
                                        2*coeff[2]*zang     + coeff[4]*xang     + coeff[5]*ymesh[m] + coeff[8]] )


                if normal[0] < 0: normal = -normal # Make sure normal points sunward
                normal = normal / np.linalg.norm(normal)
                       
            if USE_TWO_PARAMETER:
                normal = np.array( [1, -2*A*ymesh[m], -2*B*zmesh[m]] ) \
                    / np.sqrt( 1 + 4*A**2*ymesh[m]**2 + 4*B**2*zmesh[m]**2 )
                    
            if USE_ONE_PARAMETER:
                normal = np.array( [1, -2*A*ymesh[m], -2*A*zmesh[m]] ) \
                    / np.sqrt( 1 + 4*A**2*ymesh[m]**2 + 4*A**2*zmesh[m]**2 )

            # Record the angle between the normal and the x axis (1,0,0) 
            # We use this in some plots
            ang[m] = np.rad2deg( np.arccos( np.dot( normal, np.array([1,0,0]) )))

            # Create dataframe for finding boundary.  Dataframe includes data
            # on magnetosonic speed and solar wind speed needed to find bow shock
            df = create_boundary_dataframe_bs(batsrus, num_x_points, xmin, xmax, delx, 
                                           ymesh[m], zmesh[m], normal, angle)
            
            # Walk from xmax to xmin and find the first x value where magnetosonic 
            # speed is equal to solar wind speed, that is, when c_{MS} - u becomes 
            # positive.  Here, we are concerned with the component of the solar
            # wind velocity normal to the bow shock.  This will be the approximate 
            # location of the bow shock
            #
            # df[r'$c_{MS} - u_{bs\perp}$'] = df[r'$c_{MS}$'] - df[r'$u_{bs\perp}$']
            #
            # range(num_x_points-1,-1,-1) handles that ranges are xmin to xmax, and we
            # want to start at xmax
            #
            # Only do loop if u_{bs\perp} starts out supermagnetosonic.  If it
            # is submagnetosonic at the start, we'll never have a transition to 
            # submagnetosonic
            # 
            # Also only do loop if ang <75 degrees.  The bow shock disappears
            # when the normal get too large, around 80 degrees, the bow shock 
            # no longer exists.  See Basic Space Plasma Physics by Baumjohann 
            # and Treumann, 1997.
            if df[r'$c_{MS} - u_{bs\perp}$'][num_x_points-1] < 0 and ang[m] < 75.:
                for q in range(num_x_points-1,-1,-1):
                    # angNorms[m] = np.nan
                    if df[r'$c_{MS} - u_{bs\perp}$'][q] >= 0: 
                        xmesh[m] = df[r'x'][q]
                        break
            
            df.set_index(r'x', inplace=True)
        
            # Create plots that we visually inspect to determine the bow
            # shock boundary is determined
            if plotit and ((zmesh[m] < 0.01 and zmesh[m] > -0.01) or \
                           (ymesh[m] < 0.01 and ymesh[m] > -0.01)) and l > 3:
                create_2Dplots_bs(info, df, ymesh[m], zmesh[m], xmin, xmax, xmesh[m], 
                                  dtime, normal, l, angle )

        logging.info(f'Iteration: {l} Std Dev BS Diff: {np.nanstd(xmesh - xmesh_old)}')
        
        # Calculate statistics on differences between successive iterations
        # Note, no difference for the l == 0 iteration
        stdbs[l] = np.nanstd(xmesh - xmesh_old)
        
        # If the bow shock is known within tolerance (tol), exit loop
        if np.nanstd(xmesh - xmesh_old) < tol:
            break

    # Create 3D wireframe plots for the bow shock
    if angle is None:
        create_wireframe_plots_bs(info, basename, max_yz, num_yz_pts, xmesh, 
                                  ymesh, zmesh, dtime, ang, angle)
    else:
        xang = xmesh*np.cos(anglerad) + zmesh*np.sin(anglerad)
        zang = zmesh*np.cos(anglerad) - xmesh*np.sin(anglerad)
        create_wireframe_plots_bs(info, basename, max_yz, num_yz_pts, xang, 
                                  ymesh, zang, dtime, ang, angle)
    
    # Save bow shock results and stats to pickle files
    create_directory(info['dir_derived'], 'mp-bs-ns')
    pklname = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.bowshock.pkl'
    
    dfbs = pd.DataFrame()
    dfbs['y'] = ymesh
    if angle is None:
        dfbs['z'] = zmesh
        dfbs['x'] = xmesh
    else:
        dfbs['z'] = zmesh*np.cos(anglerad) - xmesh*np.sin(anglerad)
        dfbs['x'] = xmesh*np.cos(anglerad) + zmesh*np.sin(anglerad)
    dfbs['angle'] = ang
    dfbs['abs diff'] = np.abs(xmesh - xmesh_old)

    dfbs.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', pklname) )

    pklname2 = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.stats.bowshock.pkl'
    pklname3 = basename + '.' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.stats.bowshock.png'

    dfstats = pd.DataFrame()
    dfstats['STD BS'] = stdbs

    dfstats.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', pklname2) )
    
    # Also store the data in a VTK file
    xyz = ['x', 'y', 'z']
    colorvars = ['angle', 'abs diff']
   
    vtkname = basename + '-' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.bowshock'
   
    vtk_bs = sqwireframe( dfbs, xyz, colorvars, num_yz_pts )
    vtk_bs.convert_to_vtk()
    vtk_bs.write_vtk_to_file( os.path.join( info['dir_derived'], 'mp-bs-ns' ), vtkname, 'wireframe' )

    # Save data for parabolic shape
    if USE_GEN_PARAMETER:
        # Update coefficients for last iteration
        
        # Use general quadratic conic section fit for bow shock
        # a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z + k = 0 
        # and we divide by k and redefine coeficients to get
        # => a x^2 + b y^2 + c z^2 + d x y + e x z + f y z + g x + h y + i z - 1 = 0
        xn = deepcopy( xmesh.reshape(-1) )
        yn = deepcopy( ymesh )
        zn = deepcopy( zmesh )
        
        # in fit, ignore entries where we have xn = NaN
        yn = np.delete( yn, np.argwhere( np.isnan(xn) ) ) 
        zn = np.delete( zn, np.argwhere( np.isnan(xn) ) ) 
        xn = np.delete( xn, np.argwhere( np.isnan(xn) ) ) 
        
        # if angle is not None, the lines used to find the bow shock are
        # tilted with respect to x axis.  If angle is None, they are parallel 
        # to the x axis and nothing needs to be done.
        if not angle is None:
            xtmp = deepcopy( xn )
            ztmp = deepcopy( zn )
            xn = xtmp*np.cos(anglerad) + ztmp*np.sin(anglerad)
            zn = ztmp*np.cos(anglerad) - xtmp*np.sin(anglerad)

        # Use matrix math to do least squares fit to data. AA coeff = k, 
        # the eqn above in matrix form
        k = np.ones(len(xn))
        AA = np.vstack([xn**2, yn**2, zn**2, xn*yn, xn*zn, yn*zn, xn, yn, zn]).T          
        coeff = np.linalg.lstsq( AA, k, rcond=None )[0] # We only need the coefficients

        a0,b0,c0,d0,e0,f0,g0,h0,i0 = coeff
        logging.info(f'Final a:{a0}, b: {b0}, c: {c0}, d: {d0}, e: {e0}, f: {f0}, g: {g0}, h: {h0}, i: {i0}')

        # Create mesh of quadratic fit
        xmeshpara = np.zeros( ymesh.shape ) 
        angNorm = np.zeros( ymesh.shape ) 
        for m in range( len(xmeshpara) ):
           
            params = coeff.tolist()
            params.append(ymesh[m])
            params.append(zmesh[m])
            params.append(None)
            
            if np.isnan( xmesh[m] ): 
                xguess = -50. 
            else: 
                xguess = xmesh[m]

            xmeshpara[m] = fsolve( genfunc, xguess, args=params, xtol=tol )[0]

            normal = np.array( [2*coeff[0]*xmeshpara[m] + coeff[3]*ymesh[m]     + coeff[4]*zmesh[m] + coeff[6],
                                2*coeff[1]*ymesh[m]     + coeff[3]*xmeshpara[m] + coeff[5]*zmesh[m] + coeff[7],
                                2*coeff[2]*zmesh[m]     + coeff[4]*xmeshpara[m] + coeff[5]*ymesh[m] + coeff[8]] )
                
            if normal[0] < 0: normal = -normal # Make sure normal points sunward
            normal = normal / np.linalg.norm(normal)
            
            angNorm[m] = np.rad2deg( np.arccos( np.dot( normal, np.array([1,0,0]) )))
            # if angNorm[m] > 75.: xmeshpara[m] = np.NaN  # We don't trust results at large angles

    if USE_TWO_PARAMETER:
        # Update A and B coefficients for last iteration
        xn = xmesh.reshape(-1) - xbs
        yn = deepcopy( ymesh )
        zn = deepcopy( zmesh )
        yn[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN
        zn[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN

        y2n = yn**2
        z2n = zn**2
        y4n = yn**4
        z4n = zn**4
        xy2n = xn*y2n
        xz2n = xn*z2n
        y2z2n = y2n*z2n
        
        y4ns = np.nansum(y4n)
        z4ns = np.nansum(z4n)
        xy2ns = np.nansum(xy2n)
        xz2ns = np.nansum(xz2n)
        y2z2ns = np.nansum(y2z2n)
        
        denom = y4ns * z4ns - y2z2ns**2
        
        A = (z4ns * xy2ns - y2z2ns * xz2ns)/denom
        B = (y4ns * xz2ns - y2z2ns * xy2ns)/denom
        logging.info(f'Final A: {A}, B: {B}')

        # Create mesh of parabola
        xmeshpara = xbs + A * ymesh**2 + B * zmesh**2 
        # angle normal makes with x-axis, arccos of normal's dot product with (1,0,0)
        angNorm = np.rad2deg( np.arccos( 1 / np.sqrt( 1 + 4*A**2*ymesh**2 + 4*B**2*zmesh**2 ) ) ) 

    if USE_ONE_PARAMETER:
        # Update A coefficient
        xn = xmesh.reshape(-1) - xbs
        r2n = ymesh**2 + zmesh**2
        r2n[ np.argwhere( np.isnan(xn) ) ] = 0 # in sums, ignore entries where we have xn = NaN
        A = np.nansum( xn * r2n ) / np.nansum( r2n**2 )
        logging.info(f'Final A: {A}')

        # Create mesh of parabola
        xmeshpara = xbs + A * ( ymesh**2 + zmesh**2 )
        angNorm = np.rad2deg( np.arccos( 1 / np.sqrt( 1 + 4*A**2*ymesh**2 + 4*A**2*zmesh**2 ) ) ) # arccos of dot with (1,0,0))
        
    dfbspara = pd.DataFrame()
    dfbspara['y'] = ymesh
    dfbspara['z'] = zmesh
    dfbspara['x'] = xmeshpara
    dfbspara['angle'] = angNorm
    
    xyz = ['x', 'y', 'z']
    colorvars = ['angle']
   
    vtknamepara = basename + '-' + str(max_yz) + '-' + str(num_yz_pts) + '-' + str(angle) + '.bowshock-parabola'
   
    vtk_bs = sqwireframe( dfbspara, xyz, colorvars, num_yz_pts )
    vtk_bs.convert_to_vtk()
    vtk_bs.write_vtk_to_file( os.path.join( info['dir_derived'], 'mp-bs-ns' ), vtknamepara, 'wireframe' )
   
    # Create plot of convergence stats
    fig, ax = plt.subplots()   
    dfstats.plot(y=['STD BS'], title='Std Dev of Diff. Successive Iterations (Bow Shock)', 
              ylabel = 'Std Deviation', xlabel= 'Iteration',ax=ax)
    ax.axhline( y = tol, ls =":")
    fig.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', pklname3 ) )

    return dfbs

def findboundary_ns(info, filepath, time, max_xy, num_xy_pts, zlimits, num_z_points, mpfile, plotit=False):
    """Find the boundary of the neutral sheet. To find the boundary, we explore 
    a 3D grid.  The neutral sheet occurs where Bx is zero, above the neutral sheet
    Bx is in the positive x (GSM) direction and below it is in the negative x
    (GSM) direction.  So we look for when Bx is 0.
    
    To find the shape, we march down a series of lines parallel to the z axis.
    Along each line, we find where the neutral sheet is located.  
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       filepath = filepath of BATSRUS file 
       
       time = time for BATSRUS file 
       
       max_xy = grid ranges -max_xy to 0 along x axis and from -max_xy/2 to 
           +max_xy/2 along y axis (GSM)
             
       num_xy_pts = x-y plane split into num_xy_pts * num_xy_pts points to create grid (GSM)
       
       zlimits = grid ranges from zlimits[0] to zlimits[1] (GSM)

       num_z_pts = the number of points in grid along z axis (GSM)
       
       mpfile = path to magnetopause file associated with the neutral sheet
     
       plotit = Boolean, create plots for each line using create_2Dplots
           
    Outputs:
        None - results saved in pickle files and plots
    """
    
    basename = os.path.basename(filepath)
    logging.info(f'Data for BATSRUS file... {basename}')

    # Read BATSRUS file
    batsrus = swmfio.read_batsrus(filepath)
    assert(batsrus != None)
    
    # Read magnetopause pkl file
    dfmp = pd.read_pickle(os.path.join( info['dir_derived'], 'mp-bs-ns', mpfile))
    
    # Replace NaN values with large neg. values to improve interpolation.
    # NaNs in magnetopause represent values where we could not find a boundary
    xmp = np.nan_to_num( dfmp['x'], nan=-10000.)
    ymp = np.nan_to_num( dfmp['y'], nan=-10000.)
    zmp = np.nan_to_num( dfmp['z'], nan=-10000.)
    
    # Set up 2D interpolation of magnetopause data
    # Note, the reordering of x,y,z to y,z,x because x in magnetopause df is
    # a function of y,z
    interpmp = LinearNDInterpolator(list(zip(ymp, zmp)), xmp )
    
    # Convert times to datetime format for use in plots
    dtime = datetime(*time) 
    
    # Set up grid for finding bow shock and magnetopause
    
    # Look for the bow shock and magnetopause boundaries between the
    # user specified xmin and xmax
    zmin = zlimits[0]
    zmax = zlimits[1]
    delz = (zmax - zmin)/num_z_points
    
    # Create 3D mesh grid where we will record the neutral sheet surface
    # y-z grid is uniform.  x values will be updated with estimated neutral 
    # sheet positon 
    x = np.linspace(-max_xy, 0, num_xy_pts)
    y = np.linspace(-max_xy/2, max_xy/2, num_xy_pts)
    xmesh, ymesh = np.meshgrid(x, y)
    
    xmesh = xmesh.reshape(-1)
    ymesh = ymesh.reshape(-1)
    zmesh = np.full(xmesh.shape, np.nan, dtype=np.float32)

    # Use to make sure we're outside the equatorial plane of the earth
    rrmesh = xmesh**2 + ymesh**2

    # Loop thru the lines parallel to x axis that we travel down
    # The y and z offsets from the x axis are stored in ymesh and zmesh
    for m in range(len(ymesh)):
        
        # Create dataframe for finding boundaries.  Dataframe includes data
        # on magnetic pressure, dynamic ram pressure, magnetosonic speed, and
        # solar wind speed used to find boundaries
        df = create_boundary_dataframe_ns(batsrus, num_z_points, zmin, zmax, delz, 
                                        xmesh[m], ymesh[m])
        
        # Walk from zmax to zmin and find the first z value where bx is negative 
        #
        # Only do loop if bx starts out postive.  If it is negative at the start, 
        # we'll never have a transition
        if df[r'$B_x$'][num_z_points-1] > 0:
            for q in range(num_z_points-1,-1,-1):
                if df[r'$B_x$'][q] <= 0: 
                    # Verify that the point on the neutralsheet is inside the
                    # magnetopause and outside the equatorial plane of the earth
                    # behind the earth
                    xmp = interpmp( ymesh[m], df[r'z'][q] )
                    if xmesh[m] < xmp and rrmesh[m] > 1 and xmesh[m] <= 0:
                        zmesh[m] = df[r'z'][q]
                    break
        
        df.set_index(r'z', inplace=True)
        
        # Create plots that we visually inspect to determine the bow
        # shock and magnetopause boundaries, which are stored in xmeshmp
        # and xmeshbs
        if plotit:
            create_2Dplots_ns(info, df, xmesh[m], ymesh[m], zmin, zmax, zmesh[m], 
                          dtime )


    # Create 3D wireframe plots for the bow shock
    create_wireframe_plots_ns(info, basename, max_xy, num_xy_pts, xmesh, ymesh, zmesh, dtime)
    
    # Save neutral sheet results to pickle file
    create_directory(info['dir_derived'], 'mp-bs-ns')
    pklname = basename + '.' + str(max_xy) + '-' + str(num_xy_pts) + '.neutralsheet.pkl'
    
    dfns = pd.DataFrame()
    dfns['x'] = xmesh
    dfns['y'] = ymesh
    dfns['z'] = zmesh

    dfns.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', pklname) )
    
    # Also store the data in a VTK file
    xyz = ['x', 'y', 'z']
    colorvars = []
   
    vtkname = basename + '-' + str(max_xy) + '-' + str(num_xy_pts) + '.neutralsheet'
   
    vtk_mp = sqwireframe( dfns, xyz, colorvars, num_xy_pts )
    vtk_mp.convert_to_vtk()
    vtk_mp.write_vtk_to_file( os.path.join( info['dir_derived'], 'mp-bs-ns' ), vtkname, 'wireframe'  )
 
    return dfns

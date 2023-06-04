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
in the tail of the magnetosphere.
""" 

import os.path
import logging
import swmfio
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy

from deltaB import create_directory, pointcloud

# Physical constant mu0
MU0 = 1.25663706212*10**-6
    
# We assume that polytropic index, gamma = 5/3. This is the polytropic
# index of an ideal monatomic gas
GAMMA = 5/3

# In dynamic (ram) pressure equation, represents efficiency coefficient
# KAPPA = 1 is specular reflection.
KAPPA = 1

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

def create_2Dplots(info, df, yGSM, zGSM, xmin, xmax, mpline, bsline, dtime, normalmp, normalbs):
    """2D plots of parameters that change at boundaries, including solar wind
    velocity, pressure, density, and magnetic and dynamic pressures.  These plots 
    are used to confirm that bow shock and magnetopause were located. The bow
    shock (bsline) and magnetopause (mpline) are plotted on the graphs.  The bow 
    shock should occur when the solar wind speed perpendicular (u_perp)to the bow shock
    boundary goes below the magnetosonic speed (Cmbs
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
        
       df = dataframe with data to be plotted
       
       yGSM, zGSM = x and y offset for line on which bow shock and magnetopause
           locations are found.  That is, we walk along (x, yGSM, zGSM), changing
           x to find the bow shock and magnetopaus. (GSM)
       
       xmin, xmax = limits of x-axis (GSM)
       
       mpline, bsline = x axis coordinate of boundary (bow shock or magnetosphere, 
           as appropriate) (GSM)
       
       dtime = datetime of associated BATSRUS file
       
       normalmp = normal to magnetopause at the point that the line ( x[i], y, z ) 
           hits the magnetopause.  np.array of three numbers (GSM)
        
       normalbs = normal to bow shock at the point that the line ( x[i], y, z ) 
            hits the bow shock.  np.array of three numbers (GSM)

    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp_bs/')
    
    # Get angles from x axis to normals to include in titles
    angmp = np.rad2deg( np.arccos( np.dot( normalmp, np.array([1,0,0]) )))
    angbs = np.rad2deg( np.arccos( np.dot( normalbs, np.array([1,0,0]) )))
    
    # Create subplots to support multiple graphs on a page
    fig, axs = plt.subplots(6, sharex=True, figsize=(8,10))
    
    # Create plots for bow shock
    df.plot(y=[r'$u_{bs\perp}$', r'$c_{MS}$'], use_index=True, \
                ylabel = 'Speed $(km/s)$', xlabel = r'$x_{GSM}$ $(R_E)$', xlim = [xmin, xmax],
                title = 'Bow Shock ' + info['run_name'] + ' ' + str(dtime) + ' [x,' + str(yGSM) +',' + str(zGSM) + '] ' 
                    + "{:.2f}".format(bsline) + r' $R_e$ ' + "{:.2f}".format(angbs) + r'$^{\circ}$', ax=axs[0])
    df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[1])
    df.plot(y=[r'$|B|$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[2])
    df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[3])
    df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[4])
    df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = 'Density $(amu/cc)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[5])
    axs[0].axvline(x = bsline, c='black', ls=':')
    axs[1].axvline(x = bsline, c='black', ls=':')
    axs[2].axvline(x = bsline, c='black', ls=':')
    axs[3].axvline(x = bsline, c='black', ls=':')
    axs[4].axvline(x = bsline, c='black', ls=':')
    axs[5].axvline(x = bsline, c='black', ls=':')
    pltname = 'ms-u-' + str(dtime) + '-[x,' + str(yGSM) + ','  + str(zGSM) + ']'+ str([xmin, xmax]) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
        
    fig, axs = plt.subplots(6, sharex=True, figsize=(8,10))

    # Create plots for magnetopause
    df.plot(y=[r'$p_{tot}$', r'$p_{mag}$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax],
                title = 'Magnetopause ' + info['run_name'] + ' ' + str(dtime) + ' [x,' + str(yGSM) +',' + str(zGSM) + '] '\
                    + "{:.2f}".format(mpline) + r' $R_e$ ' + "{:.2f}".format(angmp) + r'$^{\circ}$', ax=axs[0])
    df.plot(y=[r'$u_x$', r'$u_y$', r'$u_z$'], use_index=True, \
                ylabel = 'Velocity $(km/s)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[1])
    df.plot(y=[r'$|B|$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[2])
    df.plot(y=[r'$B_x$', r'$B_y$', r'$B_z$'], use_index=True, \
                ylabel = 'B Field $(nT)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[3])
    df.plot(y=[r'$p$'], use_index=True, \
                ylabel = 'Pressure $(nPa)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[4])
    df.plot(y=[r'$\rho$'], use_index=True, \
                ylabel = 'Density $(amu/cc)$', xlabel = 'x_{GSM} (Re)', xlim = [xmin, xmax], ax=axs[5])
    axs[0].axvline(x = mpline, c='black', ls=':')
    axs[1].axvline(x = mpline, c='black', ls=':')
    axs[2].axvline(x = mpline, c='black', ls=':')
    axs[3].axvline(x = mpline, c='black', ls=':')
    axs[4].axvline(x = mpline, c='black', ls=':')
    axs[5].axvline(x = mpline, c='black', ls=':')
    pltname = 'ms-pp-' + str(dtime) + '-[x,' + str(yGSM) +',' + str(zGSM) + ']' + str([xmin, xmax]) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
    
    plt.close('all')
    return

def create_wireframe_plots(info, num_yz_pts, xmeshmp, xmeshbs, ymesh, zmesh, dtime, angmp, angbs):
    """ Wireframe plots of magnetopause and bow shock.  Illustrate the 3D boundaries
    
    Inputs:
       info = info on files to be processed, see info = {...} example above
       
       num_yz_pts = number of points in mesh grid = num_yz_pts x num_yz_pts x num_yz_pts
       
       xmeshmp, xmeshbs = x coordinates of wireframe for magnetopause and bow shock
           respectively (GSM)
       
       ymesh, zmesh = y and z coordinates of wireframe, the same for magnetopause
           and bow shock (GSM)
              
       dtime = datetime of BATSRUS file, used in titles and filenames
       
       angmp, angbs = angle (deg) between the x axis and the normal to magnetopause 
           (or bow shock) (GSM)
       
    Outputs:
        None = other than plots generated
    """
    create_directory(info['dir_plots'], 'mp_bs/')

    ymesh2 = ymesh.reshape((num_yz_pts,num_yz_pts))
    zmesh2 = zmesh.reshape((num_yz_pts,num_yz_pts))
    xmeshmp2 = xmeshmp.reshape((num_yz_pts,num_yz_pts))
    xmeshbs2 = xmeshbs.reshape((num_yz_pts,num_yz_pts))
    angmp2 = angmp.reshape((num_yz_pts,num_yz_pts))
    angbs2 = angbs.reshape((num_yz_pts,num_yz_pts))


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmeshmp2, ymesh2, zmesh2, label='Magnetopause')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Magnetopause')
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = 'ms-magnetopause-' + str(dtime) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(angmp2, ymesh2, zmesh2, label='Magnetopause')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Magnetopause')
    ax.set_xlabel( r'$acos( \hat n \cdot \hat x ) (Deg)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = 'ms-magnetopause-ang-' + str(dtime) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
   
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmeshbs2, ymesh2, zmesh2, label='Bow Shock')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Bow Shock' )
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = 'ms-bowshock-' + str(dtime) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(angbs2, ymesh2, zmesh2, label='Bow Shock')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Bow Shock' )
    ax.set_xlabel( r'$acos( \hat n \cdot \hat x ) (Deg)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = 'ms-bowshock-ang-' + str(dtime) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )
    
    # plt.close('all')
    return

def create_boundary_dataframe(batsrus, num_x_points, xmin, xmax, delx, y, z, normalmp, normalbs):
    """ Create dataframe based on data from BATSRUS file.  In addition to the 
    data read from the file, the dataframe includes calculated quantities to 
    determine the boundaries of the bow shock and the magnetopause.
    
    Inputs:
       batsrus = batsrus class that includes interpolator to pull values from the
           file
       
       num_x_points = the number of points along the x axis at which data will
           be interpolated (GSM)
       
       xmin, xmax = limits of interpolation along x-axis (GSM)
       
       delx = spacing between points parallel to x-axis (GSM)
       
       y, z = y and z offset.  Points will be interpolated at x[i] along x[i],y,z (GSM)
       
       normalmp = normal to magnetopause at the point that the line ( x[i], y, z ) 
           hits the magnetopause.  np.array of three numbers (GSM)

       normalbs = normal to bow shock at the point that the line ( x[i], y, z ) 
           hits the bow shock.  np.array of three numbers (GSM)
               
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
    # Ram pressure pdyn = rho * u^2 whereas dynamic pressure that is 
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
    df[r'$p_{dyn}$'] = KAPPA * rho * ((normalmp[0] * df[r'$u_x$'])**2 + 
                                      (normalmp[1] * df[r'$u_y$'])**2 + 
                                      (normalmp[2] * df[r'$u_z$'])**2) * 1.66 * 10**(-6)

    df[r'$p_{tot}$'] = df[r'$p$'] + df[r'$p_{dyn}$']
    
    df[r'$|B|$'] = np.sqrt( bx**2 + by**2 + bz**2 )
    df[r'$|u|$'] = np.sqrt( ux**2 + uy**2 + uz**2 )
    
    # Determine magnetic pressure = |normalmp x B|^2 / 2 / mu0
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
    df[r'$p_{mag}$'] = ((normalmp[1] * df[r'$B_z$'] - normalmp[2] * df[r'$B_y$'] )**2 +
                        (normalmp[2] * df[r'$B_x$'] - normalmp[0] * df[r'$B_z$'] )**2 +
                        (normalmp[0] * df[r'$B_y$'] - normalmp[1] * df[r'$B_x$'] )**2) \
                        * 10**-9 / 2 / MU0
    
    # Determine difference between dynamic and magnetic pressure, we will
    # use this to find the magnetopause.  The magnetopause is where magnetic
    # pressure equals dynamic (ram) pressure.
    # df[r'$p_{dyn} - p_{mag}$'] = df[r'$p_{dyn}$'] - df[r'$p_{mag}$']
    df[r'$p_{tot} - p_{mag}$'] = df[r'$p_{tot}$'] - df[r'$p_{mag}$']
    
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
    df[r'$u_{bs\perp}$'] = np.abs( normalbs[0] * df[r'$u_x$'] +
                             normalbs[1] * df[r'$u_y$'] +
                             normalbs[2] * df[r'$u_z$'] )
    
    # Determine difference between magnetosonic speed and solar wind speed.
    # We will use this to find the bow shock.  The bow shock is when the solar
    # wind becomes sub-magnetosonic
    df[r'$c_{MS} - u_{bs\perp}$'] = df[r'$c_{MS}$'] - df[r'$u_{bs\perp}$']
    
    return df

def initialize_xmesh( batsrus, num_x_points, xmin, xmax, delx, ymesh, zmesh):
    """Initialize the x mesh grids used in the interpolations to find the shape
    of the magnetopause and bow shock. Based on various text books and papers, we 
    assume that the magnetopause is a parabola. With the "bottom" of the parabola 
    at (x, 0, 0), and the width at the earth (0, +/-2 x, 0) and (0, 0, +/-2 x) 
    along the y and z axes respectively.  We make a similar for the bow shock.  
    From this, we initialize the x mesh grids.
    
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
        xmeshmp, xmeshbs = x mesh grids for magnetopause and bow shock (GSM)
    """

    logging.info('Initializing xmesh')

    # Create dataframe for finding boundaries.  Dataframe includes data
    # on magnetic pressures, dynamic ram pressure, magnetosonic speed, and
    # solar wind speed used to find boundaries.  We're looking nose on to the
    # boundaries, so the normals are (1,0,0) in GSM coordinates
    normalmp = np.array([1,0,0])
    normalbs = np.array([1,0,0])
    df = create_boundary_dataframe(batsrus, num_x_points, xmin, xmax, delx, 
                                   0, 0, normalmp, normalbs)

    # Walk from xmax toward xmin find the first x value where total pressure 
    # equals magnetic pressure, that is, when p_{tot} - p_{mag} becomes 
    # negative.  This will be the approximate location of the magnetopause
    #
    # df[r'$p_{tot} - p_{mag}$'] = df[r'$p_{tot}$'] - df[r'$p_{mag}$']
    #
    # range(num_x_points-1,-1,-1) handles that ranges are xmin to xmax, and we
    # want to start at xmax
    for q in range(num_x_points-1,-1,-1):
        if df[r'$p_{tot} - p_{mag}$'][q] < 0: 
            xmp = df[r'x'][q]
            break
        
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

    xmeshmp = np.zeros(ymesh.shape)
    xmeshbs = np.zeros(ymesh.shape)
    
    # Based on the paper, "Orientation and shape of the Earth's bow shock in three dimensions,"
    # V. Formisano, Planetary and Space Science, Volume 27, Issue 9, September 1979, 
    # Pages 1151-1161, assume that the magnetopause is a parabola. the width at 
    # the earth (0, +/-2 x, 0) and (0, 0, +/-2 x) along the y and z axes respectively 
    xmeshmp = xmp - 1 / 4 / xmp * ( ymesh**2 + zmesh**2 )

    # Make a similar assumption for the bow shock
    xmeshbs = xbs - 1 / 4 / xbs * ( ymesh**2 + zmesh**2 )
    
    return xmeshmp, xmeshbs

def findboundaries(info, filepath, time, max_yz, num_yz_pts, xlimits, num_x_points, maxits, tol, plotit=False):
    """Find the boundaries of the bow shock and the magnetopause. To find
    boundaries we iteratively explore a 3D grid.  The equations for the bow shock
    and magnetopause boundaries depend on the solar wind speed normal to the 
    boundary, and for the magnetopause, the magnetic field B parallel to the boundary.
    This requires us to know the shape of the boundaries to find normals.  So assume
    shapes, estimate the boundaries. And repeat iteratively until maxits or tolerance
    tol is reached.
    
    To find the shape, we march down a series of lines parallel to the x axis.
    Along each line, we find where the bow shock and the magnetopause are.  See
    create_boundary_dataframe for the boundary criteria.
    
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
    
    # Set up grid for finding bow shock and magnetopause
    
    # Look for the bow shock and magnetopause boundaries between the
    # user specified xmin and xmax
    xmin = xlimits[0]
    xmax = xlimits[1]
    delx = (xmax - xmin)/num_x_points
    
    # Find dely and delz for the gradients below
    delyz = 2 * max_yz / ( num_yz_pts - 1 )

    # Create 3D mesh grid where we will record the bow shock and magnetopause surfaces
    # y-z grid is uniform.  x values initially assume a specific boundary shape, 
    # see initialize_xmesh.  x values will be updated with estimated bow shock
    # and magnetopause locations
    y = np.linspace(-max_yz, max_yz, num_yz_pts)
    z = np.linspace(-max_yz, max_yz, num_yz_pts)
    ymesh, zmesh = np.meshgrid(y, z)
    
    ymesh = ymesh.reshape(-1)
    zmesh = zmesh.reshape(-1)
    xmeshmp, xmeshbs = initialize_xmesh( batsrus, num_x_points, xmin, xmax, delx, ymesh, zmesh)

    # Initialize counters for statistics
    to_nanmp = np.zeros(ymesh.shape, dtype=np.int16)
    from_nanmp = np.zeros(ymesh.shape, dtype=np.int16)
    to_nanbs = np.zeros(ymesh.shape, dtype=np.int16)
    from_nanbs = np.zeros(ymesh.shape, dtype=np.int16)
    
    # Initialize arrays for statistics
    stdmp = np.full(maxits, np.nan, dtype=np.float32)
    stdbs = np.full(maxits, np.nan, dtype=np.float32)
    mptonan = np.full(maxits, np.nan, dtype=np.int16)
    mpfromnan = np.full(maxits, np.nan, dtype=np.int16)
    bstonan = np.full(maxits, np.nan, dtype=np.int16)
    bsfromnan = np.full(maxits, np.nan, dtype=np.int16)

    # Do an interative process to determine the shape of magnetopause and
    # bow shock boundaries, specified by xmeshmp and xmeshbs.
    # We start with an assumed shape, recalculate boundaries.  Used recalculated 
    # boundaries to iterate again.  Rinse and repeat.
    for l in range(maxits):
        # Find the gradient, so we can get the normal to the magnetopause and bow shock
        xmeshmp2 = xmeshmp.reshape((num_yz_pts,num_yz_pts))
        xmeshbs2 = xmeshbs.reshape((num_yz_pts,num_yz_pts))
    
        # Don't forget the dy and dz
        gradmp = np.gradient(xmeshmp2, delyz)
        gradbs = np.gradient(xmeshbs2, delyz)
        
        # d/dy terms
        gradmpy = gradmp[1].reshape(-1)
        gradbsy = gradbs[1].reshape(-1)
        # d/dz terms
        gradmpz = gradmp[0].reshape(-1)
        gradbsz = gradbs[0].reshape(-1)
        
        # Keep a copy so we can see how quickly the iterations converge
        xmeshmp_old = deepcopy(xmeshmp)
        xmeshbs_old = deepcopy(xmeshbs)
        
        # Create new storage for mesh.  NaNs specify that the bow shock
        # or magnetopause (as appropriate) were not found.
        xmeshmp = np.full(ymesh.shape, np.nan, dtype=np.float32)
        xmeshbs = np.full(ymesh.shape, np.nan, dtype=np.float32)
                
        # Record the angle between the normal and the x axis (1,0,0)
        # We use this to analyze the solution.  We know that the bow shock
        # breaks down around 80 deg.  At that angle the bow shock and magnetopause
        # are oblique to the solar wind.  Thus, the solar wind normal to the
        # bow shock is so small that it is never super-magnetosonic, and hence
        # can't go sub-magnetosonic.  Similarly, the dynamic ram pressure that
        # depends on the normal component of the solar wind, is never high
        # enough to exceed the magnetic pressure.  Hence, no bow shock or 
        # magnetopause.
        angmp = np.zeros(ymesh.shape)
        angbs = np.zeros(ymesh.shape)
        
        logging.info(f'Iteration {l}...')

        # Loop thru the lines parallel to x axis that we travel down
        # The y and z offsets from the x axis are stored in ymesh and zmesh
        for m in range(len(ymesh)):
            
            # To find the magnetopause and bow shock boundaries, we need the
            # normals to the boundaries
            
            # If the gradient is NaN, we assume that we're on the flanks of the 
            # magnetopause or bow shock, as appropriate, where the boundary
            # becomes parallel to the solar wind. So we assume the normal to the
            # boundary has an angle of 85 degrees from the x axis.  Otherwise,
            # ensure that the normal is a unit vector.
            if( np.isnan( gradmpy[m] ) or np.isnan( gradmpz[m] ) ):
               rr = np.sqrt( ymesh[m]**2 + zmesh[m]**2 )
               xx = rr * np.tan( 0.0875 ) # 5 degrees 
               normalmp = np.array( [xx, ymesh[m], zmesh[m] ] ) / np.sqrt( xx**2 + rr**2 )
            else:
                normalmp = np.array( [1, -gradmpy[m], -gradmpz[m]] ) \
                    / np.sqrt( 1 + gradmpy[m]**2 + gradmpz[m]**2 )
                                        
            if( np.isnan( gradbsy[m] ) or np.isnan( gradbsz[m] ) ):
               rr = np.sqrt( ymesh[m]**2 + zmesh[m]**2 )
               xx = rr * np.tan( 0.0875 ) # 5 degrees 
               normalbs = np.array( [xx, ymesh[m], zmesh[m] ] ) / np.sqrt( xx**2 + rr**2 )
            else:
                normalbs = np.array( [1, -gradbsy[m], -gradbsz[m]] ) \
                    / np.sqrt( 1 + gradbsy[m]**2 + gradbsz[m]**2 )

            # Record the angles between the normals and the x axis (1,0,0) 
            # We use this in some plots to determine if we've covered the bow
            # shock and magnetosphere.  If we cover them, we'll see a plane of
            # constant normal of 85 degrees around the edges.
            angmp[m] = np.rad2deg( np.arccos( np.dot( normalmp, np.array([1,0,0]) )))
            angbs[m] = np.rad2deg( np.arccos( np.dot( normalbs, np.array([1,0,0]) )))

            # Create dataframe for finding boundaries.  Dataframe includes data
            # on magnetic pressure, dynamic ram pressure, magnetosonic speed, and
            # solar wind speed used to find boundaries
            df = create_boundary_dataframe(batsrus, num_x_points, xmin, xmax, delx, 
                                           ymesh[m], zmesh[m], normalmp, normalbs)
            
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
            if df[r'$p_{tot} - p_{mag}$'][num_x_points-1] > 0:
                for q in range(num_x_points-1,-1,-1):
                    # if df[r'$p_{dyn} - p_{mag}$'][q] < 0: 
                    if df[r'$p_{tot} - p_{mag}$'][q] < 0: 
                        xmeshmp[m] = df[r'x'][q]
                        break
                
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
            if df[r'$c_{MS} - u_{bs\perp}$'][num_x_points-1] < 0:
                for q in range(num_x_points-1,-1,-1):
                    if df[r'$c_{MS} - u_{bs\perp}$'][q] > 0: 
                        xmeshbs[m] = df[r'x'][q]
                        break
            
            df.set_index(r'x', inplace=True)
            
            # Increment counters for statistics
            if np.isnan(xmeshmp_old[m]) and not np.isnan(xmeshmp[m]):
                from_nanmp[m] += 1
            if not np.isnan(xmeshmp_old[m]) and np.isnan(xmeshmp[m]):
                to_nanmp[m] += 1

            if np.isnan(xmeshbs_old[m]) and not np.isnan(xmeshbs[m]):
                from_nanbs[m] += 1
            if not np.isnan(xmeshbs_old[m]) and np.isnan(xmeshbs[m]):
                to_nanbs[m] += 1

            # Create plots that we visually inspect to determine the bow
            # shock and magnetopause boundaries, which are stored in xmeshmp
            # and xmeshbs
            if l == maxits-1 and plotit:
                create_2Dplots(info, df, ymesh[m], zmesh[m], xmin, xmax, xmeshmp[m], 
                              xmeshbs[m], dtime, normalmp, normalbs )

        logging.info(f'Iteration: {l} Std Dev MP Diff: {np.nanstd(xmeshmp - xmeshmp_old)}')
        logging.info(f'Iteration: {l} Std Dev BS Diff: {np.nanstd(xmeshbs - xmeshbs_old)}')
        # logging.info(f'Iteration: {l} Total MP To Nan: {np.sum(to_nanmp)}')
        # logging.info(f'Iteration: {l} Total MP From Nan: {np.sum(from_nanmp)}')
        # logging.info(f'Iteration: {l} Total BS To Nan: {np.sum(to_nanbs)}')
        # logging.info(f'Iteration: {l} Total BS From Nan: {np.sum(from_nanbs)}')
        
        stdmp[l] = np.nanstd(xmeshmp - xmeshmp_old)
        stdbs[l] = np.nanstd(xmeshbs - xmeshbs_old)
        mptonan[l] = np.sum(to_nanmp)
        mpfromnan[l] = np.sum(from_nanmp)
        bstonan[l] = np.sum(to_nanbs)
        bsfromnan[l] = np.sum(from_nanbs)
        
        # If both the magnetosphere and bow shock are know within tolerance (tol),
        # exit loop
        if np.nanstd(xmeshmp - xmeshmp_old) < tol and np.nanstd(xmeshbs - xmeshbs_old) < tol:
            break

    # Create 3D wireframe plots for the magnetopause and the bow shock
    create_wireframe_plots(info, num_yz_pts, xmeshmp, xmeshbs, ymesh, zmesh, dtime, angmp, angbs)
    
    # Save magnetopause and bow shock results to pickle file
    create_directory(info['dir_derived'], 'mp-bs')
    pklname = basename + '.' + str(max_yz) + '.mp-bs.pkl'
    
    dfmpbs = pd.DataFrame()
    dfmpbs['y'] = ymesh
    dfmpbs['z'] = zmesh
    dfmpbs['x mp'] = xmeshmp
    dfmpbs['x bs'] = xmeshbs
    dfmpbs['angle mp'] = angmp
    dfmpbs['angle bs'] = angbs
    dfmpbs['to nan mp'] = to_nanmp
    dfmpbs['from nan mp'] = from_nanmp
    dfmpbs['to nan bs'] = to_nanbs
    dfmpbs['from nan bs'] = from_nanbs

    dfmpbs.to_pickle( os.path.join( info['dir_derived'], 'mp-bs', pklname) )

    pklname2 = basename + '.' + str(max_yz) + '.stats.mp-bs.pkl'

    dfstats = pd.DataFrame()
    dfstats['STD MP'] = stdmp
    dfstats['STD BS'] = stdbs
    dfstats['MP to NaN'] = mptonan
    dfstats['MP from NaN'] = mpfromnan
    dfstats['BS to NaN'] = bstonan
    dfstats['BS from NaN'] = bsfromnan

    dfstats.to_pickle( os.path.join( info['dir_derived'], 'mp-bs', pklname2) )
    
    fig, ax = plt.subplots()   
    dfstats.plot(y=['STD MP', 'STD BS'], title='Std Dev of Diff. Successive Iterations', 
              ylabel = 'Std Deviation', xlabel= 'Iteration',ax=ax)
    ax.axhline( y = tol, ls =":")
 
    return

def findneutralsheet(info, time, max_x, max_y, num_x_pts, num_y_pts):
    """ Find the neutral sheet in the tail region.  We use the definition
    found at https://sscweb.gsfc.nasa.gov/users_guide/ssc_reg_doc.shtml  Which
    references the Tsyganenko models (JGR, 100, 5599, 1995)
    
   Inputs:
       info = info on files to be processed, see info = {...} example above
             
       time = time for BATSRUS file 
       
       max_y = grid ranges from -max_y to +max_y along y axis (GSM)
       
       max_x = grid ranges from 0 to -max_x along x axis (GSM)
             
       num_y_pts = y split into num_y_pts to create grid
       
       num_x_pts = y split into num_y_pts to create grid
                  
    Outputs:
        
    """
    # Convert times to datetime format for use in plots
    dtime = datetime(*time) 

    # Create 3D mesh grid where we will record the bow shock and magnetopause surfaces
    # y-z grid is uniform.  x values assume a specific boundary shape, see initialize_xmesh
    x = np.linspace(-max_x, 0, num_x_pts)
    y = np.linspace(-max_y, max_y, num_y_pts)
    xmesh, ymesh = np.meshgrid(x, y)
    
    # Convert from GSM to aberrated GSM (AGSM), which is what the Tsyganenko
    # model uses.  We rotate the x-y plane 4.3 degrees = atan( 30/sqrt(400**2 + 30**2) )
    # where 400 km/s = avg solar wind and 30 km/s is speed of earth.
    # We use average values because Tsyganenko model is based on typical data.
    ang = np.deg2rad( 4.3 )
    cosang = np.cos(ang)
    sinang = np.sin(ang)
    axmesh = xmesh*cosang - ymesh*sinang
    aymesh = ymesh*cosang + xmesh*sinang

    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    from deltaB import date_timeISO

    # The dipole is aligned with z-hat in SM and dipole tilt (psi) is
    # angle between z-hat GSM
    #
    # See Magnetic Coordinate Systems
    # K.M. Laundal · A.D. Richmond, Space Sci Rev (2016) 
    ZSM = coord.Coords((0,0,1), 'SM', 'car')
    # Set time to our BATSRUS data time
    timeISO = date_timeISO( time )
    ZSM.ticks = Ticktock([timeISO], 'ISO')
    ZGSM = ZSM.convert( 'GSM', 'car' )
    psi = np.arccos( np.dot(ZGSM.data[0],(0,0,1)) )
    psi *= -1 if (ZGSM.data[0][0] < 0) else 1

    tanpsi = np.tan(psi)
    cospsi = np.cos(psi)
    sinpsi = np.cos(psi)
    
    # For the model parameters, use the values specified at
    # https://sscweb.gsfc.nasa.gov/users_guide/ssc_reg_doc.html
    RH = 8.
    delta = 4.
    G = 10.
    Ly = 10.
    
    # For each x,y pair, calculate z using the Tsyganenko model
    zmesh = 0.5 * tanpsi * ( np.sqrt( (axmesh - RH * cospsi)**2 + (delta * cospsi)**2 ) - \
                             np.sqrt( (axmesh + RH * cospsi)**2 + (delta * cospsi)**2 ) ) - \
                             G * sinpsi * aymesh**4 / (aymesh*4 + Ly**4)
 
    # Plot Neutral Sheet
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_wireframe(xmesh, ymesh, zmesh, label='Neutral Sheet')
    ax.set_title( info['run_name'] + ' ' + str(dtime) + ' Neutral Sheet')
    ax.set_xlabel( r'$X_{GSM} (R_e)$' )
    ax.set_ylabel( r'$Y_{GSM} (R_e)$' )
    ax.set_zlabel( r'$Z_{GSM} (R_e)$' )
    pltname = 'ms-neutralsheet-' + str(dtime) + '.png'
    fig.savefig( os.path.join( info['dir_plots'], 'mp_bs', pltname ) )

    # Create dataframe containing neutral sheet x,y,z mesh grid
    df = pd.DataFrame()
    df['x'] = xmesh.reshape(-1)
    df['y'] = ymesh.reshape(-1)
    df['z'] = zmesh.reshape(-1)
    df['aberrated x'] = axmesh.reshape(-1)
    df['aberrated y'] = aymesh.reshape(-1)
    
    return df


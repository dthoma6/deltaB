#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:57:20 2023

@author: Dean Thomas
"""

import deltaB as db
import logging

############################################################################
#
# Script to indentify bow shock, magnetopause, and neutral sheet boundaries
#
############################################################################

from Chigomezyo_Ngwira_092112_3a_info import info as info
from Chigomezyo_Ngwira_092112_3a_info import setup as setup

if __name__ == "__main__":
    
    # To find the bow shock, magnetopause, and neutral sheet, we explore
    # a 3D mesh grid.  We iterate over lines parallel to the x axis.  The
    # volume is defined by:
    # x-> xlimits[0] to xlimits[1]
    # y-> -max_yz to +max_yz
    # z-> -max_yz to +max_yz
    
    # Define extent of y-z plane (-max_yz to +max_yz along y and z axes)
    # mp -> magnetopause, bs -> bowshock
    max_yz_mp = 30.
    max_yz_bs = 64
    
    # Define number of points to examine = num_yz_pts**2, this defines
    # the total number of lines parallel to the x axis that we examine
    num_yz_pts_mp = 91
    num_yz_pts_bs = 65
    
    # Define number of points to sample along each line parallel to x axis
    num_x_pts = 1000
    
    # Define range of x values to examine
    xlimits = (-200,25)
    
    # Define maximum number of iterations
    maxits = 5
    
    # Define tolerance at which to stop iterations
    tol = (xlimits[1] - xlimits[0])/num_x_pts
    
    # Define extent of x-y plane (-max_xy to +max_xy along y axis and -max_xy to 0 
    # along x axis), used to find neutral sheet
    max_xy = 200.
    
    # Define number of points to examine = num_yz_pts**2, this defines
    # the total number of lines parallel to the x axis that we examine
    num_xy_pts = 201
    
    # Define number of points to sample along each line parallel to x axis
    num_z_pts = 100
    
    # Define range of z values to examine
    zlimits = (-20,20)

    # Maximum number of cores to use in multiprocessing
    maxcores = 20
    
    ###################
    # Find the bow shock and magnetosphere
    ###################

    # The times for the files that we will process, the list can be multiple times
    # for multiple files.  
    times = ((2003, 9, 2, 0, 0, 0),
              (2003, 9, 2, 1, 0, 0),
              (2003, 9, 2, 2, 0, 0),
              (2003, 9, 2, 3, 0, 0),
              (2003, 9, 2, 4, 0, 0))
    
    # Need the names of the magnetopause files, used to find the neutral sheet below
    mpfiles = ('3d__var_1_e20030902-000000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-010000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-020000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-030000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-040000-000.out.cdf.30.0-91.magnetopause.pkl')
 
    # Get a list of BATSRUS and RIM files. info parameters define location 
    # (dir_run) and file types.  Based on info structure from magnetopost.
    # NOTE: magnetopost code fails on this, so a modified version of setup
    # is in Chigomezyo_Ngwira_092112_3a_info
    setup(info)

    # Now we'll search for the magnetopause, bow shock, and neutral sheet...
    
    # When appropriate parallel process...
    if maxcores > 1:    
        # Wrapper function that contains the bulk of the routine, used
        # for parallel processing of the data
        def wrap( j, times, info, max_yz_mp, num_yz_pts_mp, max_yz_bs, num_yz_pts_bs,  
                            num_x_pts, xlimits, max_xy, num_xy_pts, zlimits, 
                            num_z_pts, mpfiles, maxits, tol ):
            # We need the filepath for BATSRUS file
            filepath = info['files']['magnetosphere'][times[j]]
         
            db.findboundary_mp(info, filepath, times[j], max_yz_mp, num_yz_pts_mp, xlimits, 
                                num_x_pts, maxits, tol, plotit=False)
            db.findboundary_bs(info, filepath, times[j], max_yz_bs, num_yz_pts_bs, xlimits, 
                                num_x_pts, maxits, tol, plotit=False)
            db.findboundary_ns(info, filepath, times[j], max_xy, num_xy_pts, zlimits, 
                                num_z_pts, mpfiles[j], plotit=False)
            
            return
    
        # Parallel process your way through the files using the wrapper function
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        Parallel(n_jobs=num_cores)(delayed(wrap)( p, times, info, 
                            max_yz_mp, num_yz_pts_mp, max_yz_bs, num_yz_pts_bs,  
                            num_x_pts, xlimits, max_xy, num_xy_pts, zlimits, 
                            num_z_pts, mpfiles, maxits, tol ) 
                                    for p in range(len(times)))
    else:
        # Loop thru the files (aka times)
        for j in range(len(times)):
            # We need the filepath for BATSRUS file
            filepath = info['files']['magnetosphere'][times[j]]
     
            db.findboundary_mp(info, filepath, times[j], max_yz_mp, num_yz_pts_mp, xlimits, 
                                num_x_pts, maxits, tol, plotit=False)
            db.findboundary_bs(info, filepath, times[j], max_yz_bs, num_yz_pts_bs, xlimits, 
                                num_x_pts, maxits, tol, plotit=False)
            db.findboundary_ns(info, filepath, times[j], max_xy, num_xy_pts, zlimits, 
                                num_z_pts, mpfiles[j], plotit=False)

        
 
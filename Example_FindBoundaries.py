#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:57:20 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db
import pandas as pd
import swmfio

#################################################################
#
# Example script for finding the bow shock, magnetopause, and
# neutral sheet based on BATSRUS data.
#
#################################################################

# info tells the script where the data files are stored and where
# to save plots and calculated data

data_dir = '/Volumes/PhysicsHDv2/runs'

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:57:20 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db
import pandas as pd
import swmfio

#################################################################
#
# Example script for generating 2D Bn (B north) versus time,
# Be (B east) versus time, and Bd (B down) versus time plots.
# Results are provided for the magnetosphere based on BATS-R-US data
# and for the gap region and the ionosphere based on RIM data  
#
#################################################################

# info tells the script where the data files are stored and where
# to save plots and calculated data

data_dir = '/Volumes/PhysicsHDv2'

info = {
        "model": "SWMF",
        "run_name": "divB_simple1",
        "rCurrents": 3.0,
        "rIonosphere": 1.01725,
        "file_type": "out",
        "dir_run": os.path.join(data_dir, "divB_simple1"),
        "dir_plots": os.path.join(data_dir, "divB_simple1.plots"),
        "dir_derived": os.path.join(data_dir, "divB_simple1.derived"),
        "dir_magnetosphere": os.path.join(data_dir, "divB_simple1", "GM"),
        "dir_ionosphere": os.path.join(data_dir, "divB_simple1", "IE")
}

if __name__ == "__main__":
    
    # To find the bow shock and  magnetopause, we explore a grid of lines.  
    # We iterate over lines parallel to the x axis.  The volume is defined by:
    # x-> xlimits[0] to xlimits[1]
    # y-> -max_yz to +max_yz
    # z-> -max_yz to +max_yz
    
    # Define extent of y-z plane (-max_yz to +max_yz along y and z axes)
    max_yz = 40.
    # Define number of points to examine = num_yz_pts**2, this defines
    # the total number of lines parallel to the x axis that we examine
    num_yz_pts = 41
    # Define number of points to sample along each line parallel to x axis
    num_x_pts = 1000
    # Define range of x values to examine
    xlimits = (-200,25)
    # Define maximum number of iterations
    maxits = 5
    # Define tolerance at which to stop iterations
    tol = (xlimits[1] - xlimits[0])/num_x_pts
    
    # TO find the neutral sheet, we explore a lines paralell to z-axis
    # Define extent of x-y plane (-max_xy/2 to +max_xy/2 along y axis and 
    # -max_xy to 0 along x axis)
    max_xy = 200.
    # Define number of points to examine = num_yz_pts**2, this defines
    # the total number of lines parallel to the x axis that we examine
    num_xy_pts = 201
    # Define number of points to sample along each line parallel to x axis
    num_z_pts = 100
    # Define range of z values to examine
    zlimits = (-20,20)

    # The times for the files that we will process, the list can be multiple times
    # for multiple files.  In this example, we have one file
    times = ( (2010, 3, 20, 1, 0, 0),)
    
    # We need the magnetopause boundary below to get the neutral sheet boundary
    # Its stored in mpfile in folder data_dir/divB_simple1.derived/mp-bs-ns
    mpfile = ('3d__mhd_4_e20100320-010000-000.out.40.0-41.magnetopause.pkl',)      
    
    # # Get a list of BATSRUS and RIM files, info parameters define location 
    # # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)

    ###########################################################
    # Find the bow shock and magnetosphere, and neutral sheet
    ###########################################################

    # Loop thru the files (aka times)
    for j in range(len(times)):
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[j]]

        db.findboundary_mp(info, filepath, times[j], max_yz, num_yz_pts, xlimits, 
                            num_x_pts, maxits, tol, plotit=False)
        db.findboundary_bs(info, filepath, times[j], max_yz, num_yz_pts, xlimits, 
                            num_x_pts, maxits, tol, plotit=False)
        db.findboundary_ns(info, filepath, times[j], max_xy, num_xy_pts, zlimits, 
                            num_z_pts, mpfile[j], plotit=False)

    # ###########################################################
    # # Convert raw SWMF files to VTK to display in Paraview
    # ###########################################################

    # for i in range(len(times)):
    #     filepath = info['files']['magnetosphere'][times[i]]
    #     batsclass = swmfio.read_batsrus(filepath)
    #     vtkfile = swmfio.write_vtk(batsclass)

  

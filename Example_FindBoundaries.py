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

data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

info = {
        "model": "SWMF",
        "run_name": "Bob_Weigel_031023_1",
        "rCurrents": 4.0,
        "rIonosphere": 1.01725,
        "file_type": "cdf",
        "dir_run": os.path.join(data_dir, "Bob_Weigel_031023_1"),
        "dir_plots": os.path.join(data_dir, "Bob_Weigel_031023_1.plots"),
        "dir_derived": os.path.join(data_dir, "Bob_Weigel_031023_1.derived"),
        "dir_ionosphere": os.path.join(data_dir, "Bob_Weigel_031023_1/IONO-2D_CDF"),
        "dir_magnetosphere": os.path.join(data_dir, "Bob_Weigel_031023_1/GM_CDF")
}    

if __name__ == "__main__":
    
    # To find the bow shock, magnetopause, and neutral sheet, we explore
    # a 3D mesh grid.  We iterate over lines parallel to the x axis.  The
    # volume is defined by:
    # x-> xlimits[0] to xlimits[1]
    # y-> -max_yz to +max_yz
    # z-> -max_yz to +max_yz
    
    # Define extent of y-z plane (-max_yz to +max_yz along y and z axes)
    max_yz = 30.
    # Define number of points to examine = num_yz_pts**2, this defines
    # the total number of lines parallel to the x axis that we examine
    num_yz_pts = 31
    # Define number of points to sample along each line parallel to x axis
    num_x_pts = 1000
    # Define range of x values to examine
    xlimits = (-200,15)
    # Define maximum number of iterations
    maxits = 10
    # Define tolerance at which to stop iterations
    tol = (xlimits[1] - xlimits[0])/num_x_pts
    
    ###################
    # Find the bow shock and magnetosphere
    ###################

    # The times for the files that we will process, the list can be multiple times
    # for multiple files.  In this example, we have one file
    # 3d__ful_4_e20100320-040000-000.out.cdf
    times = ((2010, 3, 20, 4, 0, 0),)
 
    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)

    # Loop thru the files (aka times)
    for j in range(len(times)):
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[j]]
        # base = os.path.basename(filepath)

        db.findboundaries(info, filepath, times[j], max_yz, num_yz_pts, xlimits, num_x_pts, maxits, tol)

    ###################
    # Create point clouds of neutral sheet to display in Paraview
    ###################

    base_ns = ('3d__ful_4_e20100320-040000-000.out.cdf.20.0.neutralsheet',)
    
    for i in range(len(base_ns)):
        df = db.findneutralsheet(info, times[i], 100, 30, 100, 61)

        xyz = ['x', 'y', 'z']
        colorvars = ['aberrated x', 'aberrated y']
            
        cuts = db.pointcloud( df, xyz, colorvars )
        cuts.convert_to_vtk()
        cuts.write_vtk_to_file( info['dir_derived'], base_ns[i], 'mp-bs' )
 
    ##################
    # Convert raw SWMF files to VTK to display in Paraview
    ##################

    for i in range(len(times)):
        filepath = info['files']['magnetosphere'][times[i]]
        batsclass = swmfio.read_batsrus(filepath)
        vtkfile = swmfio.write_vtk(batsclass)

    ###################
    # Create point clouds of bow shock and magnetosphere to display in Paraview
    ###################

    pklname = ('3d__ful_4_e20100320-040000-000.out.cdf.30.0.mp-bs.pkl',)
               
    base_mp = ('3d__ful_4_e20100320-040000-000.out.cdf.30.0.magnetopause',)
               
    base_bs = ('3d__ful_4_e20100320-040000-000.out.cdf.30.0.bowshock',)
     
    for i in range(len(pklname)):
        df = pd.read_pickle(os.path.join( info['dir_derived'], 'mp-bs', pklname[i]))
        
        xyz_mp = ['x mp', 'y', 'z']
        colorvars_mp = ['angle mp', 'to nan mp', 'from nan mp']
        
        xyz_bs = ['x bs', 'y', 'z']
        colorvars_bs = ['angle bs', 'to nan bs', 'from nan bs']
    
        cuts_mp = db.pointcloud( df, xyz_mp, colorvars_mp )
        cuts_mp.convert_to_vtk()
        cuts_mp.write_vtk_to_file( info['dir_derived'], base_mp[i], 'mp-bs' )
         
        cuts_bs = db.pointcloud( df, xyz_bs, colorvars_mp )
        cuts_bs.convert_to_vtk()
        cuts_bs.write_vtk_to_file( info['dir_derived'], base_bs[i], 'mp-bs' )
     


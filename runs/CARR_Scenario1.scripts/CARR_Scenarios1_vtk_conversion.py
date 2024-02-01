#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import swmfio

from CARR_Scenarios1_info import info as info

if __name__ == "__main__":
    

    # The times for the files that we will process
    TIMES = ((2019, 9, 2, 4, 15, 0),
              (2019, 9, 2, 6, 30, 0),
              (2019, 9, 2, 10, 30, 0))
  
    
    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)

    ##################
    # Convert raw SWMF files to VTK to display in Paraview
    ##################

    for i in range(len(TIMES)):
        filepath = info['files']['magnetosphere'][TIMES[i]]
        batsclass = swmfio.read_batsrus(filepath)
        vtkfile = swmfio.write_vtk(batsclass)

 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:18:39 2024

@author: Dean Thomas
"""

import logging
from numpy import ravel_multi_index

from deltaB.MHD_to_VTK import MHD_to_VTK
from deltaB.LFM_data import LFMdata

class LFM_to_VTK(MHD_to_VTK):
    """Class to convert LFM data to VTK format and to provide options 
    to save VTK file
    """
    def __init__(self, LFM):
        """Initialize LFM_to_VTK class
            
        Inputs:
            LFM = LFMClass that contains the data
                          
        Outputs:
            None
        """
        logging.info('Initializing LFM_to_VTK class') 

        # Check inputs
        assert( isinstance( LFM, LFMdata ) )
        
        # Store instance data
        self.mhd = LFM
        self.vtk_grid = None
        return
      
    def convert_to_vtk(self):
        """Convert LFM data to VTK format.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPoints, vtkUnstructuredGrid, VTK_HEXAHEDRON 
        from vtk.util import numpy_support as ns
        # from numpy import column_stack

        logging.info('Converting LFM data to VTK') 
         
        # Make sure that we have data to convert
        if( not isinstance( self.mhd, LFMdata ) ):
            logging.info('LFMClass must be specified')
            return -1
       
        # We need info on npts points and nverts vertices
        nI = self.mhd.nI
        nJ = self.mhd.nJ
        nK = self.mhd.nK
        npts = nI * nJ * nK                  # each cell has a pt at the center
        # nverts = (nI+1) * (nJ+1) * (nK+1)    # num of cell vertices
                
        # Convert xyz vertices that define cell vertices to VTK format
        cell_pts = ns.numpy_to_vtk( self.mhd.cellverticesGSM )
        vtk_points = vtkPoints()
        vtk_points.SetData(cell_pts)

        # Define unstructured grid based on the cell vertices
        if( self.vtk_grid != None ): 
            del self.vtk_grid
        self.vtk_grid = vtkUnstructuredGrid()
        self.vtk_grid.SetPoints(vtk_points)

        # Include grid structure, a stretched Cartesian grid
        # See https://examples.vtk.org/site/Python/ExplicitStructuredGrid/CreateESGrid/
        self.vtk_grid.Allocate(npts)
        for k in range(nK):
            for j in range(nJ):
                for i in range(nI):
                    multi_index = ([i, i + 1, i + 1, i, i, i + 1, i + 1, i],
                                    [j, j, j + 1, j + 1, j, j, j + 1, j + 1],
                                    [k, k, k, k, k + 1, k + 1, k + 1, k + 1])
                    pts = ravel_multi_index(multi_index, (nI+1,nJ+1,nK+1), order='F')
                    self.vtk_grid.InsertNextCell(VTK_HEXAHEDRON, 8, pts)
                 
        # Add attributes to grid, start with vectors, then scalars 
        # We use these for plotting in Paraview.  
        
        # Add vectors. '' adds cell centers, the xyz points at centers of cell
        for vv in ['b','u','j','']:
            self.add_vector_cell_attribute( vv )
       
        # Add scalars
        for sv in ['rho','V_th','p','measure']:
            self.add_scalar_cell_attribute( sv )
 
        self.vtk_grid.Modified()
        
        return 0

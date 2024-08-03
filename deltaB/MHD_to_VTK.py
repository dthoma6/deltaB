#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 09:48:40 2024

@author: Dean Thomas
"""

import logging
from deltaB.util import create_directory

class MHD_to_VTK():
    """Base class to convert MHD data to VTK format and to provide ability 
    to save VTK file.  This class is used by BATSRUS_to_VTK, OpenGGCM_to_VTK,
    etc. to provide the ability to convert MHD data to VTK format.
    """
    def __init__(self, mhd):
        

        """Initialize MHD_to_VTK class
            
        Inputs:
            mhd = MHDClass that contains the data
                          
        Outputs:
            None
        """
        logging.info('Initializing MHD_to_VTK class') 

        # Store instance data
        self.mhd = None
        self.vtk_grid = None
        return
    
    def convert_to_vtk(self):
        """Convert MHD data to VTK format.  This method must be overridden.
        Details depend upon specific MHD data.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """

        logging.info('Converting MHD data to VTK') 
 
        # Make sure that we have data to convert
        if self.mhd is None:
            logging.info('MHD data must be specified')
            return -1
       
        return -1

    def add_vector_cell_attribute( self, name ):
        """Add vector celldata attribute to grid.
         
        Inputs:
            name = name of vector variable in mhd.data_arr to be added.
                Empty string ('') is the special case to add cell centers, the
                xyz points at the center of each cell
                         
        Outputs:
            Returns -1 on err, 0 on success
        """
       
        if( self.vtk_grid == None ):
            logging.info('Before adding vector data, use create_to_vtk to create VTK grid')
            return -1
 
        from vtk.util import numpy_support as ns
        import numpy as np
        
        varidx = self.mhd.varidx
        
        varx = self.mhd.data_arr[:,varidx[name+'x']]
        vary = self.mhd.data_arr[:,varidx[name+'y']] 
        varz = self.mhd.data_arr[:,varidx[name+'z']] 
        var_data = np.column_stack((varx, vary, varz))
        var_array = ns.numpy_to_vtk( var_data )

        if name != '':
            var_array.SetName( name )
        else:
            var_array.SetName( 'cell_centers' )

        self.vtk_grid.GetCellData().AddArray( var_array )
        
        return 0  
    
    def add_scalar_cell_attribute( self, name ):
        """Add scalar celldata attribute to grid.
         
        Inputs:
            name = name of scalar variable in mhd.data_arr to be added
                         
        Outputs:
            Returns -1 on err, 0 on success
        """
       
        if( self.vtk_grid == None ):
            logging.info('Before adding scalar data, use create_to_vtk to create VTK grid')
            return -1
 
        from vtk.util import numpy_support as ns
        
        varidx = self.mhd.varidx
        var_array = ns.numpy_to_vtk( self.mhd.data_arr[:,varidx[name]] )
        var_array.SetName( name )
        self.vtk_grid.GetCellData().AddArray( var_array )
        
        return 0      

    def add_vector_point_attribute( self, name ):
        """Add vector point data attribute to grid.
         
        Inputs:
            name = name of vector variable in mhd.data_arr to be added.
                Empty string ('') is the special case to add cell centers, the
                xyz points at the center of each cell
                         
        Outputs:
            Returns -1 on err, 0 on success
        """
       
        if( self.vtk_grid == None ):
            logging.info('Before adding vector data, use create_to_vtk to create VTK grid')
            return -1
 
        from vtk.util import numpy_support as ns
        import numpy as np
        
        varidx = self.mhd.varidx
        
        varx = self.mhd.data_arr[:,varidx[name+'x']]
        vary = self.mhd.data_arr[:,varidx[name+'y']] 
        varz = self.mhd.data_arr[:,varidx[name+'z']] 
        var_data = np.column_stack((varx, vary, varz))
        var_array = ns.numpy_to_vtk( var_data )

        if name != '':
            var_array.SetName( name )
        else:
            var_array.SetName( 'cell_centers' )

        self.vtk_grid.GetPointData().AddArray( var_array )
        
        return 0  
    
    def add_scalar_point_attribute( self, name ):
        """Add scalar point data attribute to grid.
         
        Inputs:
            name = name of scalar variable in mhd.data_arr to be added
                         
        Outputs:
            Returns -1 on err, 0 on success
        """
       
        if( self.vtk_grid == None ):
            logging.info('Before adding scalar data, use create_to_vtk to create VTK grid')
            return -1
 
        from vtk.util import numpy_support as ns
        
        varidx = self.mhd.varidx
        var_array = ns.numpy_to_vtk( self.mhd.data_arr[:,varidx[name]] )
        var_array.SetName( name )
        self.vtk_grid.GetPointData().AddArray( var_array )
        
        return 0      

    def write_vtk_to_file(self, target, base, suffix):
        """Write MHD data to VTK file.
         
        Inputs:
            target = main folder that will contain subdirectory with plots
            
            base = basename of file used to create file name for plot.  
                base is derived from name of file with MHD data.
            
            suffix = suffix is used to generate file names and subdirectory.
                Plots are saved in target + suffix directory, target is the 
                overarching directory for all plots.  It contains subdirectories
                (suffix) where different types of plots are saved
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkUnstructuredGridWriter
        import os
       
        logging.info('Writing MHD VTK data to file') 
        logging.info(f'Saving {base} {suffix} VTK data')

        # Store the data in a file.
        create_directory(target, suffix +'/')
        name = base + '.vtk'
        filename = os.path.join( target, suffix, name )

        if( self.vtk_grid == None ):
            logging.info('Before saving data, use create_to_vtk to create VTK data')
            return -1
        
        if( filename == None ):
            # logging.info('Valid filename to store vtk_polydata must be provided')
            logging.info('Valid filename to store vtk_grid must be provided')
            return -1
        
        if( not filename.endswith('.vtk') ):
            logging.info('Filename ending in .vtk expected')
            return -1
        
        path = os.path.dirname(filename)
        if( not os.path.isdir(path) ):  
            logging.info('Filename must contain a path to a valid directory')
            return -1
        
        # Everything looks OK, so write data to file
        writer = vtkUnstructuredGridWriter()
        writer.SetInputData(self.vtk_grid)
        writer.SetFileName(filename)
        writer.Write()
        return 0


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:27:27 2024

@author: Dean Thomas
"""

import cdflib.cdfread as cdfread
import numpy as np
import pandas as pd
import os.path
import logging
from deltaB import create_directory, get_transform_matrix, transform

class OpenGGCMClass:
    """Class to store OpenGGCM data, follows pattern of BatsrusClass in
    swmfio for  BATSRUS data
    """
    
    def __init__(self,
                    nI        ,
                    nJ        ,
                    nK        ,
                    xGlobalMin,
                    yGlobalMin,
                    zGlobalMin,
                    xGlobalMax,
                    yGlobalMax,
                    zGlobalMax,

                    data_arr     ,
                    DataArray    ,
                    varidx       ,

                    units,
                    file):

        self.nI                = nI
        self.nJ                = nJ
        self.nK                = nK
        self.xGlobalMin        = xGlobalMin
        self.yGlobalMin        = yGlobalMin
        self.zGlobalMin        = zGlobalMin
        self.xGlobalMax        = xGlobalMax
        self.yGlobalMax        = yGlobalMax
        self.zGlobalMax        = zGlobalMax

        self.data_arr          = data_arr
        self.DataArray         = DataArray
        self.varidx            = varidx

        self.units             = units
        self.file              = file
        return

class OpenGGCM_to_VTK():
    """Class to convert OpenGGCM data to VTK format and to provide options 
    to save VTK file
    """
    def __init__(self, openggcm):
        

        """Initialize OpenGGCM_to_VTK class
            
        Inputs:
            openggcm = OpenGGCMClass that contains the data
                          
        Outputs:
            None
        """
        logging.info('Initializing OpenGGCM_to_VTK class') 

        # Check inputs
        assert( isinstance( openggcm, OpenGGCMClass ) )
        
        # Store instance data
        self.openggcm = openggcm
        self.vtk_grid = None
        return
    
    def convert_to_vtk(self):
        """Convert OpenGGCM data to VTK format.
         
        Inputs:
            None
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkPoints, vtkDoubleArray, vtkCellArray, vtkExplicitStructuredGrid, \
            vtkExplicitStructuredGridToUnstructuredGrid
        from vtk.util import numpy_support as ns

        logging.info('Converting OpenGGCM data to VTK') 
 
        # Make sure that we have data to convert
        if( not isinstance( self.openggcm, OpenGGCMClass ) ):
            logging.info('OpenGGCMClass must be specified')
            return -1
       
        # We need to info on all npts points
        nI = self.openggcm.nI
        nJ = self.openggcm.nJ
        nK = self.openggcm.nK
        npts = nI*nJ*nK
        
        varidx = self.openggcm.varidx
        xidx = varidx['x']
        yidx = varidx['y']
        zidx = varidx['z']
        
        # Convert xyz points to VTK format
        vtk_points = vtkPoints()
        vtk_cellarray = vtkCellArray()
         
        for i in range(npts):
            vtk_points.InsertNextPoint((self.openggcm.data_arr[i,xidx], 
                                    self.openggcm.data_arr[i,yidx], 
                                    self.openggcm.data_arr[i,zidx]) )

        # Include grid structure, a stretched Cartesian grid
        # See https://examples.vtk.org/site/Python/ExplicitStructuredGrid/CreateESGrid/
        for k in range(nK-1):
            for j in range(nJ-1):
                for i in range(nI-1):
                    multi_index = ([i, i + 1, i + 1, i, i, i + 1, i + 1, i],
                                   [j, j, j + 1, j + 1, j, j, j + 1, j + 1],
                                   [k, k, k, k, k + 1, k + 1, k + 1, k + 1])
                    pts = np.ravel_multi_index(multi_index, (nI,nJ,nK), order='F')
                    vtk_cellarray.InsertNextCell(8, pts)
        
        if( self.vtk_grid != None ): 
            del self.vtk_grid
        self.vtk_grid = vtkExplicitStructuredGrid()
        self.vtk_grid.SetDimensions(nI, nJ, nK)
        self.vtk_grid.SetPoints(vtk_points)
        self.vtk_grid.SetCells(vtk_cellarray)

        # Convert from vtkExplicitStructuredGrid to vtkUnstructuredGrid
        # Why?  I couldn't find the vtkExplicitStructuredGridWriter on
        # VTK website to save the grid to a file
        converter = vtkExplicitStructuredGridToUnstructuredGrid()
        converter.SetInputData(self.vtk_grid)
        converter.Update()
        self.vtk_grid = converter.GetOutput()

        # Add attributes to grid, start with vectors, then scalars 
        # We use these for plotting in Paraview
        for vv in ['b','j','u']:
            varx = self.openggcm.data_arr[:,varidx[vv+'x']]
            vary = self.openggcm.data_arr[:,varidx[vv+'y']] 
            varz = self.openggcm.data_arr[:,varidx[vv+'z']] 
            var_data = np.column_stack((varx, vary, varz))
            var_array = vtkDoubleArray()
            var_array.SetName( vv )
            var_array.SetNumberOfComponents(3)
            var_array.SetNumberOfTuples(npts)
            for x in zip(range(npts), var_data):
                var_array.SetTuple(*x)
            self.vtk_grid.GetPointData().AddArray( var_array )
            
        for sv in ['rho','p', 'eta', 'measure']:
            var_array = ns.numpy_to_vtk( self.openggcm.data_arr[:,varidx[sv]] )
            var_array.SetName( sv )
            self.vtk_grid.GetPointData().AddArray( var_array )

        self.vtk_grid.Modified()
        return 0
    

    def write_vtk_to_file(self, target, base, suffix):
        """Write OpenGGCM data to VTK file.
         
        Inputs:
            target = main folder that will contain subdirectory with plots
            
            base = basename of file used to create file name for plot.  
                base is derived from name of file with BATSRUS data.
            
            suffix = suffix is used to generate file names and subdirectory.
                Plots are saved in target + suffix directory, target is the 
                overarching directory for all plots.  It contains subdirectories
                (suffix) where different types of plots are saved
             
        Outputs:
            Returns -1 on err, 0 on success
        """
        from vtk import vtkUnstructuredGridWriter
        import os
       
        logging.info('Writing OpenGGCM VTK data to file') 
        logging.info(f'Saving {base} {suffix} VTK data')

        # Store the charts in a file.
        create_directory(target, suffix +'/')
        # filename = target + suffix + '/' + base + '.out.' + suffix + '.vtk'
        name = base + '.' + suffix + '.vtk'
        filename = os.path.join( target, suffix, name )

        # if( self.vtk_polydata == None ):
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

def get_openggcm_file_time(filepath):
    """From the cdf file "*_GM_cdf_list" read the time associated with the
    file, filepath.
     
    Inputs:
        filepath = path to CDF file that we're processing
         
    Outputs:
        Returns time associated with file as a tuple: YYYY, Month, Day, Hour,
            Minute, Second
    """
    dirname = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    basesplit = base.split('.')
    cdflist = os.path.join( dirname, basesplit[0] + '_GM_cdf_list')
    
    file = open(cdflist)
 
    lines = file.readlines()

    for line in lines:
        line = line.strip()
        linea = line.split(" ")
        if linea[0].endswith('.cdf') == False:
            continue

        datea = linea[3].split("/")
        timea = linea[7].split(":")
        time = (int(datea[0]), int(datea[1]), int(datea[2]), int(timea[0]), int(timea[1]), int(timea[2]))

        if base == linea[0]: return time

def get_openggcm_class_from_cdf(file):
    """Read OpenGGCM data from CDF file.  Store the data in OpenGGCMClass
    following the pattern used by swmfio for BATSRUS
     
    Inputs:
        file = path to CDF file
        
         
    Outputs:
        Returns OpenGGCMClass with data
    """
    # Read the file
    cdf = cdfread.CDF(file)
    globatts = cdf.globalattsget()
    
    
    # The CDF file contain four grids...
    #
    # grid_system_1 is the grid for the things that we care about.
    # That is, bx, by, bz, jz, jy, jz, ux, uy, uz, rho, p
    #
    # grid_system_2,3,4 are used for bx1, by1, and bz1, which are 
    # offset from grid_system_1.  bx1, by1, and bz1 are on the grid 
    # faces.  The x,y,z's for grid_system_2,3,4 are in x_bx, x_by, x_bz, 
    # y_bx, y_by, y_bz, z_bx, z_by, z_bz,
    nI = int(globatts['grid_system_1_dimension_1_size'])
    nJ = int(globatts['grid_system_1_dimension_2_size'])
    nK = int(globatts['grid_system_1_dimension_3_size'])
    npts = nI*nJ*nK # Total number of points in grid
    
    # Get info on variables in OpenGGCM data 
    iVar = 0
    nVar = 0
    for cdfvar in cdf.cdf_info()['zVariables']:
        if cdf.varget(cdfvar).shape == (1, npts):
            nVar += 1
    # add nVars for x,y,z,measure
    nVar += 4
    
    # Setup dicts that will conttain the list of variables and associated units
    varidx = {}
    units = {}
    
    # We'll save the OpenGGCM data in data_arr
    data_arr = np.empty((npts, nVar), dtype=np.float32);
    data_arr[:,:] = np.nan
    
    # The OpenGGCM CDF doesn't contain the full grid, just the range of
    # values for x,y,z.  We use that info to create an x,y,z grid
    # x_, y_, z_ are the ranges of values.  We loop thru them, x first,
    # then y, and finally z to fill out grid
    x_ = cdf.varget('x')[0,:]
    y_ = cdf.varget('y')[0,:]
    z_ = cdf.varget('z')[0,:]
    
    x = np.zeros(npts)
    y = np.zeros(npts)
    z = np.zeros(npts)
    for n in range(nK):
        for m in range(nJ):
            for l in range(nI):
                idx = n*nI*nJ + m*nI + l  # index current point
                x[idx] = x_[l]   
                y[idx] = y_[m]   
                z[idx] = z_[n]   

    # Get limits of grid
    xGlobalMin  = np.min(x_)
    yGlobalMin  = np.min(y_)
    zGlobalMin  = np.min(z_)
    xGlobalMax  = np.max(x_)
    yGlobalMax  = np.max(y_)
    zGlobalMax  = np.max(z_)

    # Store the x,y,z grid points.
    cdfvar = 'x'
    data_arr[:, iVar] = x
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'y'
    data_arr[:, iVar] = y
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'z'
    data_arr[:, iVar] = z
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    # Calculate the measure
    cdfvar='measure'
    # Difference between consecutive points along x,y,z axes
    dx = x_[0:-1]-x_[1:]
    dy = y_[0:-1]-y_[1:]
    dz = z_[0:-1]-z_[1:]
    
    # As with filling out the xyz grid, loop through the points - x first, y second,
    # z third - to get dx, dy, dz.  Multiply to get measure for grid rectangle
    # End points are treated separately.  
    for n in range(nK):
        if n == 0: 
            ddz = dz[0]
        elif n == nK-1:
            ddz = dz[nK-2]
        else:
            ddz = 0.5*(dz[n] + dz[n-1])

        for m in range(nJ):
            if m == 0: 
                ddy = dy[0]
            elif m == nJ-1:
                ddy = dy[nJ-2]
            else:
                ddy = 0.5*(dy[m] + dy[m-1])
                
            for l in range(nI):
                if l == 0: 
                    ddx = dx[0]
                elif l == nI-1:
                    ddx = dx[nI-2]
                else:
                    ddx = 0.5*(dx[l] + dx[l-1])
                    
                idx = n*nI*nJ + m*nI + l               # index current point
                data_arr[idx, iVar] = ddx * ddy * ddz  # calc measure 
    varidx[cdfvar] = iVar
    iVar += 1
    
    # Store the other variables stored in the CDF file
    for cdfvar in cdf.cdf_info()['zVariables']:
        # Skip bx1, by1, bz1 because they are on a different grid
        if cdfvar != 'bx1' and cdfvar != 'by1' and cdfvar != 'bz1': 
            if cdf.varget(cdfvar).shape == (1, npts):
                data_arr[:, iVar] = cdf.varget(cdfvar)[0,:]
                units[cdfvar] = cdf.varattsget(cdfvar)['units']
                varidx[cdfvar] = iVar
                iVar += 1

    # NOTE, https://openggcm.sr.unh.edu/?n=Main.Outputs
    # states "Note that the vector quantities are in "MHD" coordinates, 
    # i.e., MHD_x = - GSE_x and MHD_y = - GSE_y, MHD_z = + GSE_z." 
    data_arr[:, varidx['x']] = - data_arr[:, varidx['x']]
    data_arr[:, varidx['y']] = - data_arr[:, varidx['y']]

    data_arr[:, varidx['bx']] = - data_arr[:, varidx['bx']]
    data_arr[:, varidx['by']] = - data_arr[:, varidx['by']]

    data_arr[:, varidx['jx']] = - data_arr[:, varidx['jx']]
    data_arr[:, varidx['jy']] = - data_arr[:, varidx['jy']]

    data_arr[:, varidx['ux']] = - data_arr[:, varidx['ux']]
    data_arr[:, varidx['uy']] = - data_arr[:, varidx['uy']]
    
    # # Convert from GSE to GSM coordinates which is used elsewhere in delltaB
    # time = get_openggcm_file_time(file)
    # transmat = get_transform_matrix(time, "GSE", "GSM")
    
    # for i in range(npts):
    #     data_arr[i, varidx['x']:varidx['z']] = transform(data_arr[i, varidx['x']:varidx['z']].T, time, 'GSE', 'GSM')
    
    DataArray = data_arr.reshape((nVar, nI, nJ, nK), order='F')
    
    # Create an OpenGGCMClass to store the data, following the process
    # for BATSRUS data
    openggcmClass = OpenGGCMClass( 
                    nI          = nI,
                    nJ          = nJ,
                    nK          = nK,
                    xGlobalMin  = xGlobalMin,
                    yGlobalMin  = yGlobalMin,
                    zGlobalMin  = zGlobalMin,
                    xGlobalMax  = xGlobalMax,
                    yGlobalMax  = yGlobalMax,
                    zGlobalMax  = zGlobalMax,

                    data_arr    = data_arr,
                    DataArray   = DataArray,
                    varidx      = varidx,

                    units       = units,
                    file        = file)    
    
    return openggcmClass

if __name__ == "__main__":
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.035400.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived'
    
    oggcmclass = get_openggcm_class_from_cdf(file)
    
    tovtk = OpenGGCM_to_VTK(oggcmclass)
    tovtk.convert_to_vtk()
    
    import os.path
    basename = os.path.basename(file)
    
    tovtk.write_vtk_to_file( dir_derived, basename, 'openggcm')
    
    print('Complete')

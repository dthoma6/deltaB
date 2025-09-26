#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:27:27 2024

@author: Dean Thomas
"""

import cdflib.cdfread as cdfread
import numpy as np
import logging
import numba
from copy import deepcopy
from os import remove
from os.path import exists

from deltaB.util import get_mhd_file_time, gunzip_to_temp
from deltaB.coordinates import get_transform_matrix
from deltaB.OpenGGCM_data import OpenGGCMdata
from deltaB.OpenGGCM_curlB import OpenGGCM_curlBtoJ

####################################################################
#
# USE_CURLB, USE_FALSEB, and USE_GSE are flags for special test
# cases.  In general, all three should be FALSE
#
####################################################################

USE_CURLB = False   # Use curl of B to find current density True, 
                    # use OpenGGCM current density False
                  
USE_FALSEB = False  # Use a 'made-up' B as test case

USE_GSE = False     # When true, do not perform GSE->GSM tranformation

@numba.njit
def get_openggcm_cells_sub( xcell_, ycell_, zcell_, nI, nJ, nK):
    """ Subroutine for get_openggcm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian grid for the cells 
    
    Inputs:
        xcell_, ycell_, zcell_: x,y,z postions of cell faces on right side
            of each x_,y_,z_
        
        nI, nJ, nK: number of points along x,y,z axes in cartesian grid
        
    Returns:
        xcell, ycell, zcell numpy arrays are returned, they are the xyz 
            vertices of grid cells.
    """
    
    ncells = (nI+1) * (nJ+1) * (nK+1)
    
    xcell = np.zeros(ncells)
    ycell = np.zeros(ncells)
    zcell = np.zeros(ncells)
    
    # xcell_, ycell_, zcell_ give us the faces to the left of the grid points
    # so we need to add a face to the right of the last point
    xcellplus = np.zeros( len(xcell_)+1 )
    ycellplus = np.zeros( len(ycell_)+1 )
    zcellplus = np.zeros( len(zcell_)+1 )
    
    xcellplus[0:-1] = xcell_
    ycellplus[0:-1] = ycell_
    zcellplus[0:-1] = zcell_
    
    xcellplus[-1] = xcell_[-1] + (xcell_[-1] - xcell_[-2])
    ycellplus[-1] = ycell_[-1] + (ycell_[-1] - ycell_[-2])
    zcellplus[-1] = zcell_[-1] + (zcell_[-1] - zcell_[-2])
    
    # The OpenGGCM CDF doesn't contain the full grid, just the range of
    # values for x,y,z.  We use that info to create an x,y,z grid for the cells.
    #
    # xcell_, ycell_, zcell_ are the ranges of values.  We loop thru them, 
    # x first, then y, and finally z to fill out grid
    
    # NOTE, https://openggcm.sr.unh.edu/?n=Main.Outputs
    # states "Note that the vector quantities are in "MHD" coordinates, 
    # i.e., MHD_x = - GSE_x and MHD_y = - GSE_y, MHD_z = + GSE_z." 
    # Hence minus signs below

    for n in range(nK+1):
        for m in range(nJ+1):
           for l in range(nI+1):
               idx = n*(nI+1)*(nJ+1) + m*(nI+1) + l    # index current vertex
               xcell[idx] = -xcellplus[l]              # x,y,z of cell vertex
               ycell[idx] = -ycellplus[m]                 
               zcell[idx] =  zcellplus[n]   
    
    return xcell, ycell, zcell

@numba.njit
def get_openggcm_grid_sub( x_, y_, z_, xcell_, ycell_, zcell_, nI, nJ, nK):
    """ Subroutine for get_openggcm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian grid and the associated measures
    
    Inputs:
        x_, y_, z_: spacing between points on the cartesian axes
        
        xcell_, ycell_, zcell_: x,y,z postions of cell faces on left side
            of each x_,y_,z_
        
        nI, nJ, nK: number of points along x,y,z axes in cartesian grid
        
    Returns:
        x,y,z,measure,dx,dy,dz numpy arrays are returned
    """
    
    npts = nI*nJ*nK
    
    x = np.zeros(npts)
    y = np.zeros(npts)
    z = np.zeros(npts)
    
    measure = np.zeros(npts)
    
    dx = np.zeros(xcell_.shape)
    dy = np.zeros(ycell_.shape)
    dz = np.zeros(zcell_.shape)
    
    # Differences in cell faces used to determine cell dx,dy,dz
    dx[0:-1] = xcell_[1:] - xcell_[0:-1]
    dy[0:-1] = ycell_[1:] - ycell_[0:-1]
    dz[0:-1] = zcell_[1:] - zcell_[0:-1]
    
    dx[-1] = dx[-2]  # we only have left side faces, not the last right face
    dy[-1] = dy[-2]  # so we can't get the dx for the last point
    dz[-1] = dz[-2]  # assume last dx is the same as the next to last dx
 
    # The OpenGGCM CDF doesn't contain the full grid, just the range of
    # values for x,y,z.  We use that info to create an x,y,z grid.
    #
    # x_, y_, z_ are the ranges of values.  We loop thru them, x first,
    # then y, and finally z to fill out grid
    
    # NOTE, https://openggcm.sr.unh.edu/?n=Main.Outputs
    # states "Note that the vector quantities are in "MHD" coordinates, 
    # i.e., MHD_x = - GSE_x and MHD_y = - GSE_y, MHD_z = + GSE_z." 
    # Hence minus signs below

    # We use the same loops to determine dx, dy, dz.  Multiply dx,dy,dz to 
    # determine the measure for grid cell.  
    
    for n in range(nK):
        for m in range(nJ):
           for l in range(nI):
               idx = n*nI*nJ + m*nI + l        # index current point
               x[idx] = -x_[l]                 # x,y,z of grid pt
               y[idx] = -y_[m]                 
               z[idx] =  z_[n]   
               measure[idx] = dx[l] * dy[m] * dz[n]  # grid cell measure 
    
    return x, y, z, measure

@numba.njit
def matmul( A, B ):
    """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
    vector, allows numba accelleration
    """
    C = np.zeros(3)
    C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
    C[2] = A[2,0]*B[0] + A[2,1]*B[1] + A[2,2]*B[2]
    return C
    
@numba.njit
def transform_openggcm_variables_sub( data_arr, varidx, trans_mat):
    """Subroutine for get_openggcm_class_from_cdf that allows numba accelleration.  
    It changes the coordinate system per the transformation matrix, trans_mat.
    In this case, we're going from GSE to GSM coordinates.
    
    Inputs:
        data_arr: numpy array in which the openggcm data is stored
        
        varidx: variable indices in data_arr, e.g., x data is at varidx['x'] 
        
        trans_mat: GSE to GSM transformation matrix
                
    Returns:
        results stored in data_arr
    """
    
    xidx = varidx['x']  
    zidx = varidx['z']  
    bxidx = varidx['bx']
    bzidx = varidx['bz']
    jxidx = varidx['jx']
    jzidx = varidx['jz']
    uxidx = varidx['ux']
    uzidx = varidx['uz']
 
    for i in range(data_arr.shape[0]):
        data_arr[i, xidx:zidx+1]   = matmul( trans_mat, data_arr[i, xidx:zidx+1] )
        data_arr[i, bxidx:bzidx+1] = matmul( trans_mat, data_arr[i, bxidx:bzidx+1] )
        data_arr[i, jxidx:jzidx+1] = matmul( trans_mat, data_arr[i, jxidx:jzidx+1] )
        data_arr[i, uxidx:uzidx+1] = matmul( trans_mat, data_arr[i, uxidx:uzidx+1] )
            
    return

@numba.njit
def transform_vector_sub( vector, trans_mat):
    """Subroutine for get_openggcm_class_from_cdf that allows numba accelleration.  
    It changes the coordinate system per the transformation matrix, trans_mat.
    In this case, we're going from GSE to GSM coordinates.
    
    Inputs:
        vector: numpy array in which the vector data is stored
                
        trans_mat: GSE to GSM transformation matrix
        
        npts: total number of points in cartesian grid
        
    Returns:
        results stored in vector
    """
    for i in range(vector.shape[0]):
        vector[i, 0:3] = matmul( trans_mat, vector[i, 0:3] )
            
    return

def get_openggcm_data_from_cdf(infile, info):
    """Read OpenGGCM data from CDF file.  Store the data in OpenGGCMClass
    following the pattern used by swmfio for BATSRUS
     
    Inputs:
        infile = path to CDF file
         
    Outputs:
        Returns openggccmdata with OpenGGCM data
    """
    logging.info('Read OpenGGCM file and convert to OpenGGCMData')
    
    # Determine if this is a plain CDF or gzipped file
    if exists( infile ):
        # Its a CDF as listed in *_GM_cdf_list
        file = infile
    else:
        # Assume its a gzipped file
        file = gunzip_to_temp( infile + '.gz' )
    
    # Read the file
    cdf = cdfread.CDF(file)
    globatts = cdf.globalattsget()
    time = get_mhd_file_time(infile)
    assert( time != -1 )  # Time not found

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
    
    # Determine how many variables (nVar) in OpenGGCM data that we want to parse
    iVar = 0
    nVar = 0
    for cdfvar in cdf.cdf_info()['zVariables']:
        # Skip bx1, by1, bz1 because they are on a different grid
        if cdfvar != 'bx1' and cdfvar != 'by1' and cdfvar != 'bz1': 
            if cdf.varget(cdfvar).shape == (1, npts):
                nVar += 1
    # Add nVars for x,y,z,measure,xGSE,yGSE,zGSE,bxGSE,byGSE,bzGSE
    nVar += 10
    
    # Setup dicts that will contain the list of variables and associated units
    varidx = numba.typed.Dict.empty(key_type=numba.types.unicode_type, 
                                    value_type=numba.types.int64,)
    units  = numba.typed.Dict.empty(key_type=numba.types.unicode_type, 
                                    value_type=numba.types.unicode_type,)
    
    # The OpenGGCM CDF doesn't contain the full grid, just the range of
    # values for x,y,z.  We use that info to create an x,y,z grid.    
    logging.info('Create OpenGGCM x,y,z grid')
 
    # x_, y_, z_ are the ranges along each axis values
    xGSE_ = cdf.varget('x')[0,:]
    yGSE_ = cdf.varget('y')[0,:]
    zGSE_ = cdf.varget('z')[0,:]
        
    # Get locations of cell faces along each axis.  
    # Includes x,y,z positions of "left face" of cell
    xcellGSE_ = cdf.varget('x_bx')[0,:]
    ycellGSE_ = cdf.varget('y_by')[0,:]
    zcellGSE_ = cdf.varget('z_bz')[0,:]
    
    # Determine the x,y,z grid points and associated measures.
    # x_, y_, z_ are the ranges of values.  We loop thru them, x first,
    # then y, and finally z to fill out grid
    #
    # NOTE, https://openggcm.sr.unh.edu/?n=Main.Outputs
    # states "Note that the vector quantities are in "MHD" coordinates, 
    # i.e., MHD_x = - GSE_x and MHD_y = - GSE_y, MHD_z = + GSE_z." 
    # Hence minus signs in get_openggcm_grid_sub
    xGSE, yGSE, zGSE, measure = get_openggcm_grid_sub( xGSE_, yGSE_, zGSE_, 
                                                xcellGSE_, ycellGSE_, zcellGSE_,
                                                nI, nJ, nK )
    
    # Get limits of grid (GSE coordinates)
    xGlobalMinGSE  = np.min(-xGSE_)
    yGlobalMinGSE  = np.min(-yGSE_)
    zGlobalMinGSE  = np.min( zGSE_)
    xGlobalMaxGSE  = np.max(-xGSE_)
    yGlobalMaxGSE  = np.max(-yGSE_)
    zGlobalMaxGSE  = np.max( zGSE_)

    # Some CDF files have rCurrents, some do not
    if 'r_currents' in globatts:
        rCurrents = np.float64(globatts['r_currents'])
    else:
        rCurrents = info['rCurrents']

    # Get cells, each cell has an x,y,z point at the center and has
    # volume measure
    xcellGSE, ycellGSE, zcellGSE = get_openggcm_cells_sub( xcellGSE_, ycellGSE_, zcellGSE_,
                                                 nI, nJ, nK)
    cellverticesGSE = np.column_stack((xcellGSE, ycellGSE, zcellGSE))
    cellcentersGSE = np.column_stack((xGSE, yGSE, zGSE))

    # We'll save the OpenGGCM data in data_arr
    # data_arr = np.empty((npts, nVar), dtype=np.float32); Caused error in data processing
    data_arr = np.empty((npts, nVar))
    data_arr[:,:] = np.nan

    # We'll start with the xyz points and measures
    cdfvar = 'x'
    data_arr[:, iVar] = xGSE
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'y'
    data_arr[:, iVar] = yGSE
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'z'
    data_arr[:, iVar] = zGSE 
    units[cdfvar] = cdf.varattsget(cdfvar)['units']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar='measure'
    data_arr[:, iVar] = measure
    units[cdfvar] = cdf.varattsget('x')['units'] + '^3'
    varidx[cdfvar] = iVar
    iVar += 1
    
    # Store the other variables in the CDF file
    logging.info('Store OpenGGCM variables')
    for cdfvar in cdf.cdf_info()['zVariables']:
        # Skip bx1, by1, bz1 because they are on a different grid
        if cdfvar != 'bx1' and cdfvar != 'by1' and cdfvar != 'bz1': 
            if cdf.varget(cdfvar).shape == (1, npts):
                data_arr[:, iVar] = cdf.varget(cdfvar)[0,:]
                units[cdfvar] = cdf.varattsget(cdfvar)['units']
                varidx[cdfvar] = iVar
                iVar += 1

    # NOTE: https://openggcm.sr.unh.edu/?n=Main.Outputs
    # states "Note that the vector quantities are in "MHD" coordinates, 
    # i.e., MHD_x = - GSE_x and MHD_y = - GSE_y, MHD_z = + GSE_z." 
    # We want GSE coordinates, hence minus signs
    data_arr[:, varidx['bx']] = - data_arr[:, varidx['bx']]
    data_arr[:, varidx['by']] = - data_arr[:, varidx['by']]

    data_arr[:, varidx['jx']] = - data_arr[:, varidx['jx']]
    data_arr[:, varidx['jy']] = - data_arr[:, varidx['jy']]

    data_arr[:, varidx['ux']] = - data_arr[:, varidx['ux']]
    data_arr[:, varidx['uy']] = - data_arr[:, varidx['uy']]

    if USE_FALSEB:
        logging.info('WARNING: USE_FALSEB is True, fake B field in use. Check options')
                
        # Create a magnetic field due to a line current parallel to x-axis
        # offset 2*yGlobalMaxGSE in y-direction (So curl and div are zero inside volume)
        # yGlobalMax = yGlobalMaxGSE
        yGlobalMax = 128.0  # Make it match value in SWMF file
        
        # rho squared around x-axis
        rho2 = ( data_arr[:, varidx['y']] + 2*yGlobalMax )**2 + data_arr[:, varidx['z']]**2
        
        # New magnetic field
        data_arr[:, varidx['bx']] = 0.
        data_arr[:, varidx['by']] = - data_arr[:, varidx['z']] / rho2 # by = -sin(phi)/rho
        data_arr[:, varidx['bz']] = + (data_arr[:, varidx['y']] + 2*yGlobalMax ) / rho2 # bz = cos(phi)/rho

        # New field mean magnitude
        # Bnew = np.mean( np.sqrt(data_arr[:, varidx['bx']]**2 
        #                         + data_arr[:, varidx['by']]**2 
        #                         + data_arr[:, varidx['bz']]**2) )
        Bnew = 0.003911433508336921 # Make it match value in SWMF file
        
        # Normalize field to have a mean magnitude of Bmag
        Bmag = 20.0
        data_arr[:, varidx['bx']] = data_arr[:, varidx['bx']] * Bmag / Bnew
        data_arr[:, varidx['by']] = data_arr[:, varidx['by']] * Bmag / Bnew
        data_arr[:, varidx['bz']] = data_arr[:, varidx['bz']] * Bmag / Bnew

    if USE_CURLB or USE_FALSEB:
        logging.info('WARNING: USE_CURLB is True, check options')
        # Use curlB to determine current density, j, rather than use OpenGGCM 
        # provided values
        
        DataArray_tmp = data_arr.transpose()
        assert(np.isfortran(DataArray_tmp))
        
        DataArray_tmp = DataArray_tmp.reshape((nVar, nI, nJ, nK), order='F')
        assert(np.isfortran(DataArray_tmp))

        data_arr = OpenGGCM_curlBtoJ(data_arr, DataArray_tmp, varidx, nVar, nI, nJ, nK, rCurrents)
            
    # Keep copies of some GSE variables.  The variables above will be
    # transformed to GSM below.  We use the GSE variables in some calculations.
    
    cdfvar = 'xGSE'
    data_arr[:, iVar] = deepcopy(xGSE)
    units[cdfvar] = units['x']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'yGSE'
    data_arr[:, iVar] = deepcopy(yGSE)
    units[cdfvar] = units['y']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'zGSE'
    data_arr[:, iVar] = deepcopy(zGSE )
    units[cdfvar] = units['z']
    varidx[cdfvar] = iVar
    iVar += 1

    cdfvar = 'bxGSE'
    data_arr[:, iVar] = deepcopy(data_arr[:, varidx['bx']])
    units[cdfvar] = units['bx']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'byGSE'
    data_arr[:, iVar] = deepcopy(data_arr[:, varidx['by']])
    units[cdfvar] = units['by']
    varidx[cdfvar] = iVar
    iVar += 1
    
    cdfvar = 'bzGSE'
    data_arr[:, iVar] = deepcopy(data_arr[:, varidx['bz']] )
    units[cdfvar] = units['bz']
    varidx[cdfvar] = iVar
    iVar += 1

    # Convert to GSM coordinates
    logging.info('Convert OpenGGCM vectors from GSE to GSM coordinates')
        
    # Transformation matrix to change from GSE to GSM coordinates 
    if USE_GSE:
        # We ignor transform if USE_GSE is True
        logging.info('WARNING: USE_GSE is True, check options')
        transform_matrix     = np.identity(3) 
        rev_transform_matrix = np.identity(3) 
    else:
        transform_matrix     = get_transform_matrix(time, "GSE", "GSM", ) 
        rev_transform_matrix = get_transform_matrix(time, "GSM", "GSE", )
        
    transform_openggcm_variables_sub( data_arr, varidx, transform_matrix )
    cellverticesGSM = deepcopy( cellverticesGSE )
    transform_vector_sub( cellverticesGSM, transform_matrix )
    cellcentersGSM = deepcopy( cellcentersGSE )
    transform_vector_sub( cellcentersGSM,  transform_matrix )

    # Reshape the data array     
    DataArray = data_arr.transpose()
    assert(np.isfortran(DataArray))
    
    DataArray = DataArray.reshape((nVar, nI, nJ, nK), order='F')
    assert(np.isfortran(DataArray))
    
    # Create an instance of OpenGGCMdata to store the data, following the process
    # for BATSRUS data
    openggcmdata = OpenGGCMdata( 
                    model       = 'OpenGGCM',
                    nI          = nI,
                    nJ          = nJ,
                    nK          = nK,
                    
                    xGlobalMinGSE  = xGlobalMinGSE, # In GSE coordinates 
                    yGlobalMinGSE  = yGlobalMinGSE,  
                    zGlobalMinGSE  = zGlobalMinGSE,
                    xGlobalMaxGSE  = xGlobalMaxGSE,
                    yGlobalMaxGSE  = yGlobalMaxGSE,
                    zGlobalMaxGSE  = zGlobalMaxGSE,
                    
                    rCurrents   = rCurrents,   # scalar
                    
                    data_arr    = data_arr,    # GSM coordinates
                    DataArray   = DataArray,   # GSM coordinates
                    varidx      = varidx,
                    
                    xtickGSE    = -xGSE_,  # Ticks along x,y,z axes
                    ytickGSE    = -yGSE_,  # In GSE coordinates
                    ztickGSE    =  zGSE_,  # For minus signs, see https://openggcm.sr.unh.edu/?n=Main.Outputs
                    
                    cellcentersGSE = cellcentersGSE,   # Cell centers in GSE
                    cellcentersGSM = cellcentersGSM,   # Cell centers in GSM
                     
                    cellverticesGSE = cellverticesGSE, # GSE coordinates
                    cellverticesGSM = cellverticesGSM, # GSM coordinates

                    GSE_to_GSM  = transform_matrix,
                    GSM_to_GSE  = rev_transform_matrix,
                    
                    units       = units,
                    time        = time,
                    file        = infile)    
    
    # If necessary, delete ungzipped temporary file
    cdf.close()
    if not exists( infile ):
        remove(file)

    return openggcmdata

if __name__ == "__main__":
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.035400.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived'
    data_dir = '/Volumes/PhysicsHD/Dean_Thomas_052924_1'
    # file = '/Volumes/PhysicsHD/Dean_Thomas_020625_1/GM_CDF/Dean_Thomas_020625_1.3df.061560.cdf'
    # dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_020625_1.derived'
    # data_dir = '/Volumes/PhysicsHD/Dean_Thomas_020625_1'
    
    # Example info.  Info is used below in call to loop_ms_b
    import os
    info = {
            "model": "OpenGGCM",
            "run_name": "Dean_Thomas_052924_1",
            "file_type": "cdf",
            "rCurrents": 2.0,
            "dir_run": os.path.join(data_dir, "Dean_Thomas_052924_1"),
            "dir_plots": os.path.join(data_dir, "Dean_Thomas_052924_1.plots"),
            "dir_derived": os.path.join(data_dir, "Dean_Thomas_052924_1.derived"),
    }

    from datetime import datetime
    now = datetime.now()
    print('Start: ', now.time())
    
    oggcmdata = get_openggcm_data_from_cdf(file, info)
    
    end = datetime.now()
    print('Finish: ', end.time())
    
    _measure = oggcmdata.varidx['measure']
    measure = oggcmdata.data_arr[:,_measure]
    for i in range(len(measure)):
        assert measure[i] >= 0.
       
    from deltaB import OpenGGCM_interpolator
    
    openggcm_interp = OpenGGCM_interpolator(oggcmdata)
    openggcm_interp.register_variable( 'bx' )
    xxGSM = np.array( [-10,-12,15] )
    bx = openggcm_interp.interpolator(xxGSM, 'bx')[0]
    print(bx)

    from deltaB import convert_mhd_to_dataframe, create_deltaB_spherical_dataframe
    
    df = convert_mhd_to_dataframe( oggcmdata )
    df = create_deltaB_spherical_dataframe( df )

    from deltaB.OpenGGCM_to_VTK import OpenGGCM_to_VTK
    
    tovtk = OpenGGCM_to_VTK(oggcmdata)
    tovtk.convert_to_vtk()
    
    import os.path
    basename = os.path.basename(file)
    
    tovtk.write_vtk_to_file( dir_derived, basename, 'vtk')
    
    complete = datetime.now()
    print('Complete: ', complete.time())


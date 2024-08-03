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

from deltaB.util import get_mhd_file_time
from deltaB.coordinates import get_transform_matrix
from deltaB.LFM_data import LFMdata
from copy import deepcopy

@numba.njit
def get_lfm_cylindrical_sub(x, y, z, nIp1, nJp1, nKp1):
    """ Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian cell centers in cylindrical coordinates

    Inputs:
        x, y, z: x,y,z postions of cell center points

        nIp1, nJp1, nKp1: number of cell center points

    Returns:
        xcenter, rcenter, acenter numpy arrays are returned
    """
    
    x_ = x.reshape(nKp1,nJp1,nIp1)
    y_ = y.reshape(nKp1,nJp1,nIp1)
    z_ = z.reshape(nKp1,nJp1,nIp1)
    
    # LFM grid is distorted spherical coordinate system
    # useful to examine cylindrical coordinates x, r, azimuth

    # xcenter is one slice in azimuth, xcenter identical across slices
    xcenter = x_[0,:,:]
    
    # rcenter is one slice in azimuth, rcenter identical acroos slices
    r_ = np.sqrt(y_**2 + z_**2)
    rcenter = r_[0,:,:]

    # acenter contains the various azimuth values, one per azimuth slice
    # We want acenter to be from 0->2pi, not -pi->pi
    a_ = np.arctan2(z_, y_)
    acenter = np.zeros(nKp1)
    for i in range(nKp1):
        if a_[i,0,0] >= 0:
            acenter[i] = a_[i,0,0]
        else:
            acenter[i] = 2*np.pi + a_[i,0,0]          

    # Return the transposes to get them in the same form as DataArray
    return xcenter.T, rcenter.T, acenter.T

@numba.njit
def get_lfm_cell_sub(x, y, z, nIp1, nJp1, nKp1):
    """ Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian cell vertices

    Inputs:
        x, y, z: x,y,z postions of cell center points

        nIp1, nJp1, nKp1: number of cell center points

    Returns:
        xcell, ycell, zcell, measure (dV) numpy arrays are returned
    """
    # Constant used below.  How far do we go outside the simulation boundary
    # is determined by FACTOR
    FACTOR = 1.1 

    nIp2 = nIp1 + 1
    nJp2 = nJp1 + 1
    nKp2 = nKp1 + 1
    
    x_ = x.reshape(nKp1,nJp1,nIp1)
    y_ = y.reshape(nKp1,nJp1,nIp1)
    z_ = z.reshape(nKp1,nJp1,nIp1)
    
    xcell = np.zeros((nKp2,nJp2,nIp2))
    ycell = np.zeros((nKp2,nJp2,nIp2))
    zcell = np.zeros((nKp2,nJp2,nIp2))
    measure = np.zeros((nKp1,nJp1,nIp1))
    
    # LFM grid is distorted spherical coordinate system
    # useful to examine cylindrical coordinates x, r, azimuth

    a_ = np.arctan2(z_, y_)
    r_ = np.sqrt(y_**2 + z_**2)

    # In cylinderical coordinates with shape nKp1,nJp1,nIp1, azimuth pages are 
    # constant, e.g., 1st page all 0 deg, 2nd page all 3.1 deg, 3rd page 6.2 deg, ...
    # Each x page is identical.  Each r page is identical.  That is, same x's and
    # r's examined on each azimuth slice.

    # To determine cell vertices, set up the common x and r pages.  
    xvert = np.zeros((nJp2,nIp2))
    rvert = np.zeros((nJp2,nIp2))
    
    xvert[0   ,0:-1] = 1.5*x_[0,0,:] - 0.5*x_[0,1,:]   # 1st row is before initial x 
    xvert[1:-1,0:-1] = 0.5*(x_[0,0:-1,:] + x_[0,1:,:]) # Most are the midpts of neighboring x's
    xvert[-1  ,0:-1] = 1.5*x_[0,-1,:] - 0.5*x_[0,-2,:] # Last row beyond x
    xvert[:   ,  -1] = FACTOR*xvert[:,-2]              # Last column repeats
    
    rvert[0   ,0:-1] = 1.5*r_[0,0,:] - 0.5*r_[0,1,:]   # 1st row behind initial x 
    rvert[1:-1,0:-1] = 0.5*(r_[0,0:-1,:] + r_[0,1:,:]) # most are the avg of neighboring x's
    rvert[-1  ,0:-1] = 1.5*r_[0,-1,:] - 0.5*r_[0,-2,:] # Last row beyond x
    rvert[:   ,  -1] = FACTOR*rvert[:,-2]              # Last column repeats
    
    # Since azimuth is constant on each page, we use simple array for azimuth.
    # One entry per page
    avert = np.zeros(nKp2)
    atmp  = np.zeros(nKp1)
    
    # For filling in avert below, we want a_ to be from 0->2pi, not -pi->pi
    for i in range(nKp1):
        if a_[i,0,0] >= 0:
            atmp[i] = a_[i,0,0]
        else:
            atmp[i] = 2*np.pi + a_[i,0,0]          
        
    avert[0]    = 0.5*(atmp[0] + atmp[-1])             # Most are midpts of
    avert[1:-2] = 0.5*(atmp[0:-2] + atmp[1:-1])        # neighboring slices in az
    avert[-2]   = 0.5*(2*np.pi + atmp[-2] + atmp[-1])  # Azimuth wraps around here
    avert[-1]   = 0.5*(atmp[0] + atmp[-1])
        
    # Get x,y,z of each vertex
    for i in range(nKp2):
        xcell[i,:,:] = xvert
        ycell[i,:,:] = rvert * np.cos(avert[i])
        zcell[i,:,:] = rvert * np.sin(avert[i])
    
    xcell = xcell.reshape(-1)
    ycell = ycell.reshape(-1)
    zcell = zcell.reshape(-1)
    
    # Determine delta x, r, azimuth to calculate measure
    # Note: ignore last column which is a repeat, see above
    dxvert = np.abs(xvert[1:,0:-1] - xvert[0:-1,0:-1])
    drvert = np.abs(rvert[1:,0:-1] - rvert[0:-1,0:-1])
    
    davert = np.abs(avert[1:] - avert[0:-1])
    davert[-1] = np.abs(avert[-1] - (avert[-2] - 2*np.pi)) # Azimuth wraps around

    # Calculate measure in culindrical coordinates
    for i in range(nKp1):
        measure[i,:,:] = dxvert * drvert * davert[i]
    
    measure = measure.reshape(-1)
    
    return xcell, ycell, zcell, measure
 
@numba.njit
def matmul(A, B):
    """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
    vector, allows numba accelleration
    """
    C = np.zeros(3)
    C[0] = A[0, 0]*B[0] + A[0, 1]*B[1] + A[0, 2]*B[2]
    C[1] = A[1, 0]*B[0] + A[1, 1]*B[1] + A[1, 2]*B[2]
    C[2] = A[2, 0]*B[0] + A[2, 1]*B[1] + A[2, 2]*B[2]
    return C

@numba.njit
def transform_lfm_variables_sub(data_arr, varidx, trans_mat, npts):
    """Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It changes the coordinate system per the transformation matrix, trans_mat.
    In this case, we're going from SM to GSM coordinates.

    Inputs:
        data_arr: numpy array in which the openggcm data is stored

        varidx: variable indices in data_arr, e.g., x data is at varidx['x'] 

        trans_mat: SM to GSM transformation matrix

        npts: total number of points in cartesian grid

    Returns:
        results stored in data_arr
    """

    xidx = varidx['x']
    zidx = varidx['z']
    bxidx = varidx['bx']
    bzidx = varidx['bz']
    # jxidx = varidx['jx']
    # jzidx = varidx['jz']
    uxidx = varidx['ux']
    uzidx = varidx['uz']

    for i in range(npts):
        data_arr[i,   xidx:zidx+1] = matmul(trans_mat, data_arr[i, xidx:zidx+1])
        data_arr[i, bxidx:bzidx+1] = matmul(trans_mat, data_arr[i, bxidx:bzidx+1])
        # data_arr[i, jxidx:jzidx+1] = matmul( trans_mat, data_arr[i, jxidx:jzidx+1] )
        data_arr[i, uxidx:uzidx+1] = matmul(trans_mat, data_arr[i, uxidx:uzidx+1])

    return

@numba.njit
def transform_vector_sub( vector, trans_mat):
    """Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It changes the coordinate system per the transformation matrix, trans_mat.
    In this case, we're going from SM to GSM coordinates.
    
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

def get_lfm_data_from_cdf(file):
    """Read LFM data from CDF file.  Store the data in OpenGGCMClass
    following the pattern used by swmfio for BATSRUS

    Inputs:
        file = path to CDF file

    Outputs:
        Returns lfmdata with LFM data
    """
    logging.info('Read LFM file and convert to LFMData')

    # Read the file
    cdf = cdfread.CDF(file)
    globatts = cdf.globalattsget()
    time = get_mhd_file_time(file)
    assert (time != -1)  # Time not found

    # Determine size of grid
    grid = globatts['grid']     # info on grid dimensions is in a string
    grida = grid.split()        # so we parse it
    gridb = grida[2].split('x')
    nI = int(gridb[0])          # nI,nJ,nK gives number of cells
    nJ = int(gridb[1])
    nK = int(gridb[2])
    nIp1 = nI + 1            
    nJp1 = nJ + 1
    nKp1 = nK + 1
    ncells = nIp1 * nJp1 * nKp1 # Total number of cells in grid

    # Determine how many variables (nVar) in LFM data that we want to parse
    iVar = 0
    nVar = 0
    for cdfvar in cdf.cdf_info()['zVariables']:
        # skip kameleon_identity_unknown_*
        if not cdfvar.startswith('kameleon'):
            if cdf.varget(cdfvar).shape == (1, ncells):
                nVar += 1
    # Add nVars for measure
    nVar += 1

    # Setup dicts that will contain the list of variables and associated units
    varidx = numba.typed.Dict.empty(key_type=numba.types.unicode_type,
                                    value_type=numba.types.int64,)
    units = numba.typed.Dict.empty(key_type=numba.types.unicode_type,
                                   value_type=numba.types.unicode_type,)

    # We'll store the LFM data in data_arr
    data_arr = np.empty((ncells, nVar), dtype=np.float32)
    data_arr[:, :] = np.nan

    logging.info('Store LFM variables')
    for cdfvar in cdf.cdf_info()['zVariables']:
        # skip kameleon_identity_unknown_*
        if not cdfvar.startswith('kameleon'):
            if cdf.varget(cdfvar).shape == (1, ncells):
                data_arr[:, iVar] = cdf.varget(cdfvar)[0, :]
                units[cdfvar] = cdf.varattsget(cdfvar)['units']
                varidx[cdfvar] = iVar
                iVar += 1
                
    # Units in CDF files are incorrect
    # Distances are in cm not Re
    data_arr[:, varidx['x']] = data_arr[:, varidx['x']] / 100.0 / 1000.0 / 6378.1
    data_arr[:, varidx['y']] = data_arr[:, varidx['y']] / 100.0 / 1000.0 / 6378.1
    data_arr[:, varidx['z']] = data_arr[:, varidx['z']] / 100.0 / 1000.0 / 6378.1
    # B is in Gauss not nT
    data_arr[:, varidx['bx']] = data_arr[:, varidx['bx']] * 100000.
    data_arr[:, varidx['by']] = data_arr[:, varidx['by']] * 100000.
    data_arr[:, varidx['bz']] = data_arr[:, varidx['bz']] * 100000.
    # u is in cm/sec not km/sec
    data_arr[:, varidx['ux']] = data_arr[:, varidx['ux']] / 100000.
    data_arr[:, varidx['uy']] = data_arr[:, varidx['uy']] / 100000.
    data_arr[:, varidx['uz']] = data_arr[:, varidx['uz']] / 100000.
    
    logging.info('Create LFM x,y,z grid')

    # x,y,z in Re in SM.  We need SM for some calculations, but x,y,z in 
    # data_arr will be converted to GSM below.  
    xSM = cdf.varget('x')[0, :] / 100.0 / 1000.0 / 6378.1  
    ySM = cdf.varget('y')[0, :] / 100.0 / 1000.0 / 6378.1
    zSM = cdf.varget('z')[0, :] / 100.0 / 1000.0 / 6378.1

    # Get limits of grid
    xGlobalMinSM = np.min(xSM)
    yGlobalMinSM = np.min(ySM)
    zGlobalMinSM = np.min(zSM)
    xGlobalMaxSM = np.max(xSM)
    yGlobalMaxSM = np.max(ySM)
    zGlobalMaxSM = np.max(zSM)
    rCurrents = np.float32(globatts['r_currents'])

    # Get cell vertices and measure. Each cell has a point xSM, ySM, zSM at the center
    # We also need cell centers x, r, and azimuth (cylindrical coordiantes) for interpolation
    xcellSM, ycellSM, zcellSM, measure = get_lfm_cell_sub( xSM, ySM, zSM, nIp1, nJp1, nKp1)
    xcenterSM, rcenterSM, acenterSM = get_lfm_cylindrical_sub( xSM, ySM, zSM, nIp1, nJp1, nKp1)

    cdfvar='measure'
    data_arr[:, iVar] = measure
    units[cdfvar] = cdf.varattsget('x')['units'] + '^3'
    varidx[cdfvar] = iVar
    iVar += 1

    cellverticesSM = np.column_stack((xcellSM, ycellSM, zcellSM))
    cellcentersSM  = np.column_stack((xSM, ySM, zSM))

    # Transform to GSM coordiantes
    logging.info('Convert LFM vectors from SM to GSM coordinates')

    transform_matrix = get_transform_matrix(time, "SM", "GSM", )
    transform_lfm_variables_sub( data_arr, varidx, transform_matrix, ncells)
    
    cellverticesGSM = deepcopy( cellverticesSM )
    transform_vector_sub( cellverticesGSM, transform_matrix )
    
    cellcentersGSM = deepcopy( cellcentersSM )
    transform_vector_sub( cellcentersGSM,  transform_matrix )

    # Reshape the data array
    DataArray = data_arr.transpose()
    assert(np.isfortran(DataArray))

    DataArray = DataArray.reshape((nVar, nI+1, nJ+1, nK+1), order='F')
    assert(np.isfortran(DataArray))

    # Create an instance of LFMdata to store the data, following the process
    # for BATSRUS data
    lfmdata = LFMdata(
                    model       = 'LFM',
                    nI          = nI,
                    nJ          = nJ,
                    nK          = nK,

                    xGlobalMinSM  = xGlobalMinSM,
                    yGlobalMinSM  = yGlobalMinSM,
                    zGlobalMinSM  = zGlobalMinSM,
                    xGlobalMaxSM  = xGlobalMaxSM,
                    yGlobalMaxSM  = yGlobalMaxSM,
                    zGlobalMaxSM  = zGlobalMaxSM,
    
                    rCurrents     = rCurrents,

                    data_arr    = data_arr,
                    DataArray   = DataArray,
                    varidx      = varidx,

                    cellcentersSM    = cellcentersSM,
                    cellcentersGSM   = cellcentersGSM,
                    cellverticesSM   = cellverticesSM,
                    cellverticesGSM  = cellverticesGSM,
                    
                    xcenterSM   = xcenterSM,
                    rcenterSM   = rcenterSM,
                    acenterSM   = acenterSM,

                    SM_to_GSM   = transform_matrix,
                    GSM_to_SM   = get_transform_matrix(time, "GSM", "SM", ),

                    units       = units,
                    time        = time,
                    file        = file)

    return lfmdata

if __name__ == "__main__":
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_2/GM_CDF/null.LFM.Dean_Thomas_052924_2_mhd_2000-01-01T02-20-00Z.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_2.derived'

    from datetime import datetime
    now = datetime.now()
    print('Start: ', now.time())

    lfmdata = get_lfm_data_from_cdf(file)

    end = datetime.now()
    print('Finish: ', end.time())

    # from deltaB import convert_mhd_to_dataframe, create_deltaB_spherical_dataframe

    # df = convert_mhd_to_dataframe( lfmdata )
    # df = create_deltaB_spherical_dataframe( df )

    # from deltaB.LFM_to_VTK import LFM_to_VTK

    # tovtk = LFM_to_VTK(lfmdata)
    # tovtk.convert_to_vtk()

    # import os.path
    # basename = os.path.basename(file)

    # tovtk.write_vtk_to_file( dir_derived, basename, 'vtk')

    # complete = datetime.now()
    # print('Complete: ', complete.time())
    
    # illustrate structure of lfm grid
    import matplotlib.pyplot as plt

    xcellcenter_ = lfmdata.xcenterSM
    rcellcenter_ = lfmdata.rcenterSM
    acellcenter_ = lfmdata.acenterSM
    
    # slices in azimuth are identical when looking at cylinderical coordinates

    # azimuth range
    plt.plot(acellcenter_[:]*180/np.pi, '.')
    plt.ylabel( 'Azimuth (degree)')
    plt.xlabel( 'Slice number')
    plt.title('Azimuth wraps on last slice')
    plt.show()

    # full x,r grid on one azimuth slice
    for i in range(xcellcenter_.shape[0]):
        plt.plot(xcellcenter_[i,:], rcellcenter_[i,:], '.')
    plt.xlabel( 'x (Re)')
    plt.ylabel( 'r (Re)')
    plt.title('Full grid on one azimuth slice')
    plt.show()

    # one distorted semicircle on one azimuth slice
    plt.plot(xcellcenter_[100,:], rcellcenter_[100,:],'.')
    plt.xlabel( 'x (Re)')
    plt.ylabel( 'r (Re)')
    plt.title('Single distorted semicircle')
    plt.show()

    # outer boundary
    plt.plot(xcellcenter_[-1,:], rcellcenter_[-1,:],'.')
    plt.xlabel( 'x (Re)')
    plt.ylabel( 'r (Re)')
    plt.title('Outer Boundary is a cylinder')
    plt.show()







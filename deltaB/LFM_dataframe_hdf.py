#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:27:27 2024

@author: Dean Thomas
"""

# import cdflib.cdfread as cdfread
from pyhdf.SD import SD, SDC
import numpy as np
import logging
import numba

from deltaB.util import get_mhd_file_time2
from deltaB.coordinates import get_transform_matrix
from deltaB.LFM_data import LFMdata
from deltaB.LFM_curl import lfm_curl
import deltaB.CISM.jcalc2ptr as jcalc2ptr
from copy import deepcopy

# True if we use jcalc2ptr to calculate curlB to get current density
# False we use our own code
####################################################################
# Do NOT use jcalc2ptr, it does not take into account that the cell
# normals do not necessarily coincide with xhat, yhat, zhat.  So we 
# need to change to the xhat, yhat, zhat basis
####################################################################
USE_JCALC2PTR = False
assert not USE_JCALC2PTR

def get_lfm_current_sub( cdf, measure, nI, nJ, nK ):
    """ Subroutine for get_lfm_class_from_cdf. It determines the MHD
    current density

    Inputs:
        cdf: cdf object that contains data from file
        
        measure: size (dV) of each grid cell
        
        nI, nJ, nK: number of grid centers(x,y,z)
        
    Returns:
        jx, jy, jz numpy arrays are returned
    """
    
    # We need to calculate the current density, which is not provided in the
    # CDF file.  We use jcalc2ptr to do the calculation.  The routine is referenced
    # at https://wiki.ucar.edu/display/LTR/Output as Pjcalc2.F.  The code is on 
    # the CISM website https://cism.hao.ucar.edu/cismdx/
    
    # For the call to jcalc2ptr, we'll need the grid vertices
    data_xyz = np.empty( ((nI+1)*(nJ+1)*(nK+1), 3), dtype=np.float32)
    data_xyz[:, 0] = cdf.varget('x')[0, :]
    data_xyz[:, 1] = cdf.varget('y')[0, :]
    data_xyz[:, 2] = cdf.varget('z')[0, :]
   
    DAxyz = data_xyz.T
    DAxyz = DAxyz.reshape(3,nI+1,nJ+1,nK+1)

    x_ = DAxyz[0,:,:,:]
    y_ = DAxyz[1,:,:,:]
    z_ = DAxyz[2,:,:,:]
 
    # We also need bx, by, bz, but we need to throw away the extraneous extra
    # data.  Only nI*nJ*nK is valid.  
    # See https://wiki.ucar.edu/display/LTR/Output
    
    rawdata = cdf.varget('bx')[0, :]
    rawdata = rawdata.reshape([nK+1,nJ+1,nI+1])
    bx_ = rawdata[0:nK,0:nJ,0:nI].T

    rawdata = cdf.varget('by')[0, :]
    rawdata = rawdata.reshape([nK+1,nJ+1,nI+1])
    by_ = rawdata[0:nK,0:nJ,0:nI].T

    rawdata = cdf.varget('bz')[0, :]
    rawdata = rawdata.reshape([nK+1,nJ+1,nI+1])
    bz_ = rawdata[0:nK,0:nJ,0:nI].T
    
    # We need the measure, but in a larger [nI+1, nJ+1, nK+1] arrat
    measure_ = np.zeros([nI+1, nJ+1, nK+1])
    measure_[:,:,:] = np.nan
    measure_[0:nI,0:nJ,0:nK] = measure.reshape([nK,nJ,nI]).T 

    # Use CISM jcalc2ptr to determine the current density
    current = jcalc2ptr.jcalc2ptr( bx_, by_, bz_, x_, y_, z_, measure_, nI, nJ, nK, 
              nI+1, nJ+1, nK+1, nJ+2 )

    ###################################################################
    ###################################################################
    #
    # jcalc2ptr provides results in A/m^2, but we want microAmp/m^2
    # so multiple by 10^6.
    #
    # jcalc2ptr uses Stokes theorem, from the line integral we get:
    #
    # J = 1/mu0 Curl B => 1/mu0 B dl/dA
    #
    # 1/mu0 B dl/dA = 1/4pi/10^-7 [m/H] B [Gauss] dl[cm]/dA[cm^2]
    #               with 1 Gauss = 10^-4 Tesla and 100 cm = 1 m
    #               = 1/4pi 10^7 10^-4 [m/H] [T]/[cm]
    #               = 1/4pi 10^7 10^-2 [m/H] [T]/[m]
    #               with H = kg m^2/(s^2 A^2) and T = kg m/(s^2 A m)
    #               = 1/4pi 10^7 10^-2 [m s^2 A^2/(kg m^2)] [kg m/(s^2 A m)]/[m]
    #               = 1/4pi 10^7 10^-2 [A/m^2]
    #               = 1/4pi 10^7 10^-2 10^6 [microAmp/m^2]
    #               = 1/4pi 10^11 [microAmp/m^2] 
    #               = 7.9577471 * 10^9
    #
    # But jcalc2ptr uses 7.9577471 * 10^3, see line 40 of jcalc2ptr.F
    # so we multiply by 10^6 to get microAmps/m^2.  Consistent with BATSRUS 
    # and OpenGGCM
    #
    ####################################################################
    ####################################################################     
    
    current = current * 10**6
    
    # Return proper parts of current array, see jcalc2ptr
    jx = current[0,:,1:-1,0:-1]
    jy = current[1,:,1:-1,0:-1]
    jz = current[2,:,1:-1,0:-1]
        
    return jx, jy, jz

@numba.njit
def norm(x):
    return np.sqrt( x[0]**2 + x[1]**2 + x[2]**2 )

@numba.njit
def get_lfm_current_sub2( gridx, gridy, gridz, bx, by, bz, nI, nJ, nK ):
    """ Subroutine for get_lfm_class_from_cdf. It determines the MHD
    current density using Stokes Theorem to calculate curl B.  Stokes Theorem
    tells us integral over surface of curl B dot dA = integral over line integral
    of B dot dl.  We calculate the line integral to determine curl B

    Inputs:
        gridx, gridy, gridz: x,y,z grid points from cdf file.  We don't want the
            cdf object because it breaks numba
        
        bx,by,bz = B field from cdf file
        
        nI,nJ,nK = dimensions of cell centers
        
    Returns:
        jx, jy, jz numpy arrays are returned
    """
    
    # We need to calculate the current density, which is not provided in the
    # CDF file.  

    # We'll need the grid vertices
    grid = np.zeros(((nI+1)*(nJ+1)*(nK+1), 3))
    grid[:, 0] = gridx
    grid[:, 1] = gridy
    grid[:, 2] = gridz
    grid = grid.T
    grid = grid.reshape( 3, nI+1, nJ+1, nK+1 )
    
    # and the magnetic field
    b = np.zeros(((nI+1)*(nJ+1)*(nK+1), 3))
    b[:, 0] = bx
    b[:, 1] = by
    b[:, 2] = bz
    b = b.T
    b = b.reshape( 3, nI+1, nJ+1, nK+1 )
    
    # Storage for current density results
    jx = np.zeros((nI, nJ, nK))
    jy = np.zeros((nI, nJ, nK))
    jz = np.zeros((nI, nJ, nK))
    
    ###################################################################
    ###################################################################
    #
    # We want current density in microAmp/m^2 so multiply by 7.9577471 * 10^9
    #
    # To calc curlB, we use Stokes theorem.  From the line integral we get:
    #
    # J = 1/mu0 curl B => 1/mu0 B dl/dA
    #
    # 1/mu0 B dl/dA = 1/4pi/10^-7 [m/H] B [Gauss] dl[cm]/dA[cm^2]
    #               with 1 Gauss = 10^-4 Tesla and 100 cm = 1 m
    #               = 1/4pi 10^7 10^-4 [m/H] [T]/[cm]
    #               = 1/4pi 10^7 10^-2 [m/H] [T]/[m]
    #               with H = kg m^2/(s^2 A^2) and T = kg m/(s^2 A m)
    #               = 1/4pi 10^7 10^-2 [m s^2 A^2/(kg m^2)] [kg m/(s^2 A m)]/[m]
    #               = 1/4pi 10^7 10^-2 [A/m^2]
    #               = 1/4pi 10^7 10^-2 10^6 [microAmp/m^2]
    #               = 1/4pi 10^11 [microAmp/m^2] 
    #               = 7.9577471 * 10^9
    #
    ####################################################################
    ####################################################################     

    # Loop through each cell in the grid and evaluate the curl of B
    # Remember the last rows of data are bad, so go to nI-1,nJ-1,nK-1
    for i in range(nI-1):
        for j in range(nJ-1):
            for k in range(nK-1):
                (jx[i,j,k], jy[i,j,k], jz[i,j,k]) = 7.9577471 * 10**9 * lfm_curl( grid, b, i, j, k )
    
    # Fill in missing entries from loops above.  Loops only go to nI-2, etc.
    jx[:,:,nK-1] = jx[:,:,nK-2]
    jy[:,:,nK-1] = jy[:,:,nK-2]
    jz[:,:,nK-1] = jz[:,:,nK-2]

    jx[:,nJ-1,:] = jx[:,nJ-2,:]
    jy[:,nJ-1,:] = jy[:,nJ-2,:]
    jz[:,nJ-1,:] = jz[:,nJ-2,:]

    jx[nI-1,:,:] = jx[nI-2,:,:]
    jy[nI-1,:,:] = jy[nI-2,:,:]
    jz[nI-1,:,:] = jz[nI-2,:,:]
    
    # # Excessive large values 
    # jx[:,0,:] = jx[:,1,:] #0 
    # jy[:,0,:] = jy[:,1,:] #0
    # jz[:,0,:] = jz[:,1,:] #0

    # # Excessive large values 
    # jx[0,:,:] = jx[1,:,:] #0 
    # jy[0,:,:] = jy[1,:,:] #0
    # jz[0,:,:] = jz[1,:,:] #0

    # # Excessive large values 
    # jx[:,:,0] = jx[:,:,1] #0 
    # jy[:,:,0] = jy[:,:,1] #0
    # jz[:,:,0] = jz[:,:,1] #0

    return jx, jy, jz

@numba.njit
def get_lfm_cylindrical_sub(x, y, z, nI, nJ, nK):
    """ Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian cell centers in cylindrical coordinates

    Inputs:
        x, y, z: x,y,z postions of cell centers

        nI, nJ, nK: number of cell centers

    Returns:
        xcenter, rcenter, acenter numpy arrays are returned
    """
    
    x_ = x.reshape(nK,nJ,nI)
    y_ = y.reshape(nK,nJ,nI)
    z_ = z.reshape(nK,nJ,nI)
    
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
    acenter = np.zeros(nK)
    for i in range(nK):
        # Use average to cancel out noise introduce finding cell centers
        aavg = np.average( a_[i,1:nJ,1:nI] )          
        if aavg >= 0:
            acenter[i] = aavg
        else:
            acenter[i] = 2*np.pi + aavg         

    # Return the transposes to get them in the same form as DataArray
    return xcenter.T, rcenter.T, acenter.T

@numba.njit
def get_lfm_cellcenters_sub(xcell, ycell, zcell, nI, nJ, nK):
    """ Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian cell centers

    Inputs:
        xcell, ycell, zcell: x,y,z postions of cell vertices

        nI, nJ, nK: nI*nJ*nK gives number of cells

    Returns:
        x, y, z, measure (dV) numpy arrays are returned
    """
    nIp1 = nI + 1  # Needed to get number of cell vertices
    nJp1 = nJ + 1
    nKp1 = nK + 1
    
    xcell_ = xcell.reshape(nKp1,nJp1,nIp1)
    ycell_ = ycell.reshape(nKp1,nJp1,nIp1)
    zcell_ = zcell.reshape(nKp1,nJp1,nIp1)
    
    x = np.zeros((nK,nJ,nI))
    y = np.zeros((nK,nJ,nI))
    z = np.zeros((nK,nJ,nI))
    measure = np.zeros((nK,nJ,nI))
        
    # Cell centers are midpoints
    x = 0.5*(xcell_[0:-1,0:-1,0:-1] + xcell_[0:-1,0:-1,1:  ])
    y = 0.5*(ycell_[0:-1,0:-1,0:-1] + ycell_[0:-1,1:  ,0:-1])
    z = 0.5*(zcell_[0:-1,0:-1,0:-1] + zcell_[1:  ,0:-1,0:-1])
    
    # Determine cell measure, aka volumes
    for i in range(nI):
        for j in range(nJ):
            for k in range(nK):
                # Based on average triple product of vectors.  x0 and x1 are
                # diagonals on one face x2 is length.  Area is 1/2 cross product 
                # of diagonals. Volume is 1/2 x2 dot (x1 cross x0)
                x0 = np.array( [ xcell_[k  ,j+1,i+1] - xcell_[k  ,j  ,i  ], 
                                 ycell_[k  ,j+1,i+1] - ycell_[k  ,j  ,i  ], 
                                 zcell_[k  ,j+1,i+1] - zcell_[k  ,j  ,i  ] ] )
                
                x1 = np.array( [ xcell_[k  ,j+1,i  ] - xcell_[k  ,j  ,i+1], 
                                 ycell_[k  ,j+1,i  ] - ycell_[k  ,j  ,i+1], 
                                 zcell_[k  ,j+1,i  ] - zcell_[k  ,j  ,i+1] ] )
                
                x2 = np.array( [ xcell_[k+1,j  ,i  ] - xcell_[k  ,j  ,i  ], 
                                 ycell_[k+1,j  ,i  ] - ycell_[k  ,j  ,i  ], 
                                 zcell_[k+1,j  ,i  ] - zcell_[k  ,j  ,i  ] ] )
                
                # Similarly, x3 and x4 are the diagonals on opposite face, 
                # x5 is length
                x3 = np.array( [ xcell_[k+1,j+1,i+1] - xcell_[k+1,j  ,i  ], 
                                 ycell_[k+1,j+1,i+1] - ycell_[k+1,j  ,i  ], 
                                 zcell_[k+1,j+1,i+1] - zcell_[k+1,j  ,i  ] ] )
                
                x4 = np.array( [ xcell_[k+1,j+1,i  ] - xcell_[k+1,j  ,i+1], 
                                 ycell_[k+1,j+1,i  ] - ycell_[k+1,j  ,i+1], 
                                 zcell_[k+1,j+1,i  ] - zcell_[k+1,j  ,i+1] ] )
                
                x5 = np.array( [ xcell_[k+1,j+1,i+1] - xcell_[k  ,j+1,i+1], 
                                 ycell_[k+1,j+1,i+1] - ycell_[k  ,j+1,i+1], 
                                 zcell_[k+1,j+1,i+1] - zcell_[k  ,j+1,i+1] ] )
                
                # Measure is average of two volumes
                measure[k,j,i] = 0.25*( np.abs( np.dot( x2, np.cross(x1, x0) ) ) +
                                        np.abs( np.dot( x5, np.cross(x4, x3) ) ) )
                
                assert measure[k,j,i] >= 0
                
    x = x.reshape(-1)    
    y = y.reshape(-1)    
    z = z.reshape(-1) 
    measure = measure.reshape(-1)
    
    return x, y, z, measure
 
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
    jxidx = varidx['jx']
    jzidx = varidx['jz']
    uxidx = varidx['ux']
    uzidx = varidx['uz']

    for i in range(npts):
        data_arr[i,   xidx:zidx+1] = matmul(trans_mat, data_arr[i, xidx:zidx+1])
        data_arr[i, bxidx:bzidx+1] = matmul(trans_mat, data_arr[i, bxidx:bzidx+1])
        data_arr[i, jxidx:jzidx+1] = matmul(trans_mat, data_arr[i, jxidx:jzidx+1])
        data_arr[i, uxidx:uzidx+1] = matmul(trans_mat, data_arr[i, uxidx:uzidx+1])

    return

@numba.njit
def transform_vector_sub( vector, trans_mat ):
    """Subroutine for get_lfm_class_from_cdf that allows numba accelleration.  
    It changes the coordinate system per the transformation matrix, trans_mat.
    In this case, we're going from SM to GSM coordinates.
    
    Inputs:
        vector: numpy array in which the vector data is stored
                
        trans_mat: SM to GSM transformation matrix
        
        npts: total number of points in cartesian grid
        
    Returns:
        results stored in vector
    """
    for i in range(vector.shape[0]):
        vector[i, 0:3] = matmul( trans_mat, vector[i, 0:3] )
            
    return

def get_lfm_data_from_hdf(file):
    """Read LFM data from CDF file.  Store the data in LFMClass
    following the pattern used by swmfio for BATSRUS

    Inputs:
        file = path to CDF file

    Outputs:
        Returns lfmdata with LFM data
    """
    logging.info('Read LFM file and convert to LFMData')

    # Read the file
    # cdf = cdfread.CDF(file)
    # globatts = cdf.globalattsget()
    hdf = SD(file, SDC.READ)

    time = get_mhd_file_time2(file)
    assert (time != -1)  # Time not found

    # Determine size of grid
    # grid = globatts['grid']     # info on grid dimensions is in a string
    # grida = grid.split()        # so we parse it
    # gridb = grida[2].split('x')
    # nI = int(gridb[0])          
    # nJ = int(gridb[1])
    # nK = int(gridb[2])
    X_grid = hdf.select('X_grid')
    nI = X_grid.attributes()['ni'] - 1
    nJ = X_grid.attributes()['nj'] - 1
    nK = X_grid.attributes()['nk'] - 1
    # nIp1 = nI + 1            
    # nJp1 = nJ + 1
    # nKp1 = nK + 1
    # ncellverts = nIp1 * nJp1 * nKp1 # Total number of cell vertices in grid
    npts = nI * nJ * nK             # Total number of data points in grid

    # Determine how many variables (nVar) in LFM data that we want to parse
    # ['X_grid', 'Y_grid', 'Z_grid', 'bx_', 'by_', 'bz_', 'vx_', 'vy_', 'vz_', 'rho_', 'c']
    nVar = 11
    # nVar = 0
    # ['X_grid', 'Y_grid', 'Z_grid', 'bx_', 'by_', 'bz_', 'vx_', 'vy_', 'vz_', 'rho_', 'c']
    # for cdfvar in cdf.cdf_info()['zVariables']:
    #     # skip kameleon_identity_unknown_* and other unused variables
    #     if cdfvar in ['x', 'y', 'z', 'bx', 'by', 'bz', 'ux', 'uy', 'uz', 'rho', 'V_th']:
    #         if cdf.varget(cdfvar).shape == (1, ncellverts):
    #             nVar += 1
  
    # Add nVars for measure, current density, and pressure
    nVar += 5

    # Setup dicts that will contain the list of variables and associated units
    varidx = numba.typed.Dict.empty(key_type=numba.types.unicode_type,
                                    value_type=numba.types.int64,)
    units = numba.typed.Dict.empty(key_type=numba.types.unicode_type,
                                   value_type=numba.types.unicode_type,)

    # iVar = 0
    # for cdfvar in cdf.cdf_info()['zVariables']:
    #     # skip kameleon_identity_unknown_* and other unused variables
    #     # if cdfvar in ['x', 'y', 'z', 'bx', 'by', 'bz', 'ux', 'uy', 'uz', 'rho', 'V_th']:
    #         if cdf.varget(cdfvar).shape == (1, ncellverts):
    #             units[cdfvar] = cdf.varattsget(cdfvar)['units']
    #             varidx[cdfvar] = iVar
    #             iVar += 1
    iVar = 0

    units['x'] = 'R'
    varidx['x'] = iVar
    iVar += 1
        
    units['y'] = 'R'
    varidx['y'] = iVar
    iVar += 1
        
    units['z'] = 'R'
    varidx['z'] = iVar
    iVar += 1
        
    units['bx'] = 'nT'
    varidx['bx'] = iVar
    iVar += 1
        
    units['by'] = 'nT'
    varidx['by'] = iVar
    iVar += 1
        
    units['bz'] = 'nT'
    varidx['bz'] = iVar
    iVar += 1
        
    units['ux'] = 'km/s'
    varidx['ux'] = iVar
    iVar += 1
        
    units['uy'] = 'km/s'
    varidx['uy'] = iVar
    iVar += 1
        
    units['uz'] = 'km/s'
    varidx['uz'] = iVar
    iVar += 1
        
    units['rho'] = 'amu/cm^3'
    varidx['rho'] = iVar
    iVar += 1
        
    units['V_th'] = 'cm/s'
    varidx['V_th'] = iVar
    iVar += 1
                  
    logging.info('Create LFM x,y,z grid')

    # Find cell centers and measure (dV) based on cell vertices
    # xvertexSM = cdf.varget('x')[0, :] 
    # yvertexSM = cdf.varget('y')[0, :] 
    # zvertexSM = cdf.varget('z')[0, :] 
    X_grid = hdf.select('X_grid')
    Y_grid = hdf.select('Y_grid')
    Z_grid = hdf.select('Z_grid')
    xx = np.array(X_grid.get())
    yy = np.array(Y_grid.get())
    zz = np.array(Z_grid.get())
    xvertexSM = xx.reshape(-1)
    yvertexSM = yy.reshape(-1)
    zvertexSM = zz.reshape(-1)

    xcenterSM, ycenterSM, zcenterSM, measure = get_lfm_cellcenters_sub( xvertexSM, 
                                                                        yvertexSM, 
                                                                        zvertexSM, 
                                                                        nI, nJ, nK)

    # We'll save these in SM coordinates in the LFMdata object
    cellverticesSM = np.column_stack((xvertexSM, yvertexSM, zvertexSM))
    cellcentersSM  = np.column_stack((xcenterSM, ycenterSM, zcenterSM))

    # The LFM grid is divided into azimuth slices, with the cylindrical
    # x and r coordinates identical on each slice.  Get x and r plus
    # the azimuth values (SM coordinates).  Useful in interpolation routines
    xsliceSM, rsliceSM, asliceSM = get_lfm_cylindrical_sub( xcenterSM, 
                                                            ycenterSM, 
                                                            zcenterSM, 
                                                            nI, nJ, nK)

    # Get limits of grid (SM coordinates)
    xGlobalMinSM = np.min(xcenterSM)
    yGlobalMinSM = np.min(ycenterSM)
    zGlobalMinSM = np.min(zcenterSM)
    xGlobalMaxSM = np.max(xcenterSM)
    yGlobalMaxSM = np.max(ycenterSM)
    zGlobalMaxSM = np.max(zcenterSM)
    # rCurrents    = np.float32(globatts['r_currents'])
    rCurrents    = 2.5

    ###################################################################
    ###################################################################
    #
    # Here we get rid of the meaningless data on the last indices
    # 
    # As described on https://wiki.ucar.edu/display/LTR/Output
    #
    # "We think about the LFM grid as being of size NI, NJ, NK where N? describes 
    # the number of cells in a given logical direction. So for the 53x24x32 grid, 
    # there are ni+1, nj+1 and nk+1 points. All of the data sets in the file are 
    # of size NIP1 x NJP1 x NKP1 where "P1" stands for "plus 1". The "plus 1" size 
    # comes from defining the edges of the cells.
    #
    # Some variables are cell-centered, while others are aligned on grid edges 
    # & faces. We define all the arrays to be the same size for coding simplicity 
    # within the LFM. Note: this means that certain variables are not defined & 
    # contain meaningless data on the last indices!"
    #
    ###################################################################
    ###################################################################
    
    logging.info('Clean and store LFM data')

    # We'll store the LFM data in data_pts that is at cell centers
    data_pts = np.empty((npts, nVar), dtype=np.float32)
    data_pts[:, :] = np.nan

    data_pts[:, varidx['x']] = xcenterSM
    data_pts[:, varidx['y']] = ycenterSM
    data_pts[:, varidx['z']] = zcenterSM
    
    hdfvar='measure'
    data_pts[:, iVar] = measure
    units[hdfvar] = 'R^3'
    varidx[hdfvar] = iVar
    iVar += 1

    raw0 = hdf.select('bx_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['bx']] = data.reshape(-1)
    
    raw0 = hdf.select('by_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['by']] = data.reshape(-1)
    
    raw0 = hdf.select('bz_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['bz']] = data.reshape(-1)
    
    raw0 = hdf.select('vx_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['ux']] = data.reshape(-1)
    
    raw0 = hdf.select('vy_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['uy']] = data.reshape(-1)
    
    raw0 = hdf.select('vz_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['uz']] = data.reshape(-1)
    
    raw0 = hdf.select('rho_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['rho']] = data.reshape(-1)
    
    raw0 = hdf.select('c_')
    raw1 = np.array(raw0.get())
    data = raw1[0:nK,0:nJ,0:nI]
    data_pts[:, varidx['V_th']] = data.reshape(-1)

    # Current density isn't in the CDF, we have to calculate it
    X_grid = hdf.select('X_grid')
    Y_grid = hdf.select('Y_grid')
    Z_grid = hdf.select('Z_grid')
    xx = np.array(X_grid.get())
    yy = np.array(Y_grid.get())
    zz = np.array(Z_grid.get())
    gridx = xx.reshape(-1)
    gridy = yy.reshape(-1)
    gridz = zz.reshape(-1)

    # gridx = cdf.varget('x')[0, :]
    # gridy = cdf.varget('y')[0, :]
    # gridz = cdf.varget('z')[0, :]
    
    bx_grid = hdf.select('bx_')
    by_grid = hdf.select('by_')
    bz_grid = hdf.select('bz_')
    xx = np.array(bx_grid.get())
    yy = np.array(by_grid.get())
    zz = np.array(bz_grid.get())
    bx = xx.reshape(-1)
    by = yy.reshape(-1)
    bz = zz.reshape(-1)

    # bx = cdf.varget('bx')[0, :]
    # by = cdf.varget('by')[0, :]
    # bz = cdf.varget('bz')[0, :]

    # if USE_JCALC2PTR:
    #     jx,  jy, jz = get_lfm_current_sub( cdf, measure, nI, nJ, nK )
    # else:
    #     jx, jy, jz = get_lfm_current_sub2( gridx, gridy, gridz, bx, by, bz, nI, nJ, nK )
    jx, jy, jz = get_lfm_current_sub2( gridx, gridy, gridz, bx, by, bz, nI, nJ, nK )
                
    varidx['jx'] = iVar
    data_pts[:,varidx['jx']] = (jx.T).reshape(-1)
    units['jx'] = 'muA/m^2'
    iVar += 1
    
    varidx['jy'] = iVar
    data_pts[:,varidx['jy']] = (jy.T).reshape(-1)
    units['jy'] = 'muA/m^2'
    iVar += 1
    
    varidx['jz'] = iVar
    data_pts[:,varidx['jz']] = (jz.T).reshape(-1)
    units['jz'] = 'muA/m^2'
    iVar += 1
        
    # Thermal pressure isn't in the CDF, we have to calculate it
    # To convert from rho * V_th^2 = g/cm^3 * cm^2/s^2
    #                              = g/cm/s^2 * 1 kg/ 1000 g * 100 cm/m
    #                              = 0.1 kg/m/s^2 
    #                              = 0.1 Pa
    #                              = 10^8 nPa
    varidx['p'] = iVar
    data_pts[:,varidx['p']] = data_pts[:,varidx['rho']] * data_pts[:,varidx['V_th']]**2 * 10**8
    units['p'] = 'nPa'
    iVar += 1
    
    # Reshape the data array following BATSRUS process 
    DataPts = data_pts.transpose()
    assert(np.isfortran(DataPts))
    
    DataPts = DataPts.reshape((nVar, nI, nJ, nK), order='F')
    assert(np.isfortran(DataPts))

    # Units in HDF files are not what we want
    # x,y,z, is in cm not Re
    cellverticesSM = cellverticesSM / 100.0 / 1000.0 / 6378.1 
    cellcentersSM  = cellcentersSM  / 100.0 / 1000.0 / 6378.1 
    xsliceSM       = xsliceSM       / 100.0 / 1000.0 / 6378.1 
    rsliceSM       = rsliceSM       / 100.0 / 1000.0 / 6378.1 
    xGlobalMinSM   = xGlobalMinSM   / 100.0 / 1000.0 / 6378.1
    yGlobalMinSM   = yGlobalMinSM   / 100.0 / 1000.0 / 6378.1
    zGlobalMinSM   = zGlobalMinSM   / 100.0 / 1000.0 / 6378.1
    xGlobalMaxSM   = xGlobalMaxSM   / 100.0 / 1000.0 / 6378.1
    yGlobalMaxSM   = yGlobalMaxSM   / 100.0 / 1000.0 / 6378.1
    zGlobalMaxSM   = zGlobalMaxSM   / 100.0 / 1000.0 / 6378.1
    
    data_pts[:, varidx['x']] = data_pts[:, varidx['x']] / 100.0 / 1000.0 / 6378.1
    data_pts[:, varidx['y']] = data_pts[:, varidx['y']] / 100.0 / 1000.0 / 6378.1
    data_pts[:, varidx['z']] = data_pts[:, varidx['z']] / 100.0 / 1000.0 / 6378.1
    
    data_pts[:, varidx['measure']] = data_pts[:, varidx['measure']] / ( 100.0 * 1000.0 * 6378.1 )**3
    
    # B is in Gauss not nT
    data_pts[:, varidx['bx']] = data_pts[:, varidx['bx']] * 100000.
    data_pts[:, varidx['by']] = data_pts[:, varidx['by']] * 100000.
    data_pts[:, varidx['bz']] = data_pts[:, varidx['bz']] * 100000.
    
    # u is in cm/sec not km/sec
    data_pts[:, varidx['ux']] = data_pts[:, varidx['ux']] / 100000.
    data_pts[:, varidx['uy']] = data_pts[:, varidx['uy']] / 100000.
    data_pts[:, varidx['uz']] = data_pts[:, varidx['uz']] / 100000.
    
    # rho is in g/cm^3 not amu/cm^3
    data_pts[:, varidx['rho']] = data_pts[:, varidx['rho']] * 6.022 * 10**23

    # Transform to GSM coordinates
    logging.info('Convert LFM vectors from SM to GSM coordinates')

    transform_matrix = get_transform_matrix(time, "SM", "GSM", )
    transform_lfm_variables_sub( data_pts, varidx, transform_matrix, npts)
    
    cellverticesGSM = deepcopy( cellverticesSM )
    transform_vector_sub( cellverticesGSM, transform_matrix )
    
    cellcentersGSM = deepcopy( cellcentersSM )
    transform_vector_sub( cellcentersGSM,  transform_matrix )
    
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

                    data_arr    = data_pts,
                    DataArray   = DataPts,
                    varidx      = varidx,

                    cellcentersSM    = cellcentersSM,
                    cellcentersGSM   = cellcentersGSM,
                    cellverticesSM   = cellverticesSM,
                    cellverticesGSM  = cellverticesGSM,
                    
                    xsliceSM   = xsliceSM,
                    rsliceSM   = rsliceSM,
                    asliceSM   = asliceSM,
                    
                    SM_to_GSM   = transform_matrix,
                    GSM_to_SM   = get_transform_matrix(time, "GSM", "SM", ),

                    units       = units,
                    time        = time,
                    file        = file)
    
    return lfmdata

if __name__ == "__main__":
    # file = '/Volumes/PhysicsHD/Dean_Thomas_052924_2/GM_CDF/null.LFM.Dean_Thomas_052924_2_mhd_2000-01-01T00-20-00Z.cdf'
    # file = '/Volumes/PhysicsHD/Dean_Thomas_052924_2/GM_CDF/null.LFM.Dean_Thomas_052924_2_mhd_2000-01-01T18-20-00Z.cdf'
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_2/GM_HDF/Dean_Thomas_052924_2_mhd_2000-01-01T00-20-00Z.hdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_2.derived'

    from datetime import datetime
    now = datetime.now()
    print('Start: ', now.time())

    lfmdata = get_lfm_data_from_hdf(file)

    end = datetime.now()
    print('Finish: ', end.time())
    
    # from deltaB import convert_mhd_to_dataframe, create_deltaB_spherical_dataframe

    # df = convert_mhd_to_dataframe( lfmdata )
    # df = create_deltaB_spherical_dataframe( df )

    from deltaB.LFM_to_VTK import LFM_to_VTK

    tovtk = LFM_to_VTK(lfmdata)
    tovtk.convert_to_vtk()

    import os.path
    basename = os.path.basename(file)

    tovtk.write_vtk_to_file( dir_derived, basename, 'vtk')

    complete = datetime.now()
    print('Complete: ', complete.time())
    
    # # Code below compares jcalc2ptr to my current density code
    # # jcalc2ptr does not take into account that the cell faces
    # # may not be aligned with xhat, yhat, zhat
    
    # DIST = 10.
    # NUM = 10
    # gridn = np.zeros((3,NUM,NUM,NUM))
    # B    = np.zeros((3,NUM,NUM,NUM))
    # measure = np.zeros((NUM,NUM,NUM))
    
    # nI = NUM-1
    # nJ = NUM-1
    # nK = NUM-1
    
    # for i in range(NUM):
    #     for j in range(NUM):
    #         for k in range(NUM):
    #             # B = [y, x, z], with curlB = [-1,-1,-1]
    #             gridn[0,i,j,k] = i*DIST
    #             gridn[1,i,j,k] = j*DIST
    #             gridn[2,i,j,k] = k*DIST
    #             measure[i,j,k] = DIST*DIST*DIST
                
    # # Test that we get the correct curlB when we randomly rotate the cube of points
    
    # if True:
    #     import random
    #     A = np.zeros((3,3))
        
    #     anum = random.randint(0,90)
    #     bnum = random.randint(0,90)
    #     gnum = random.randint(0,90)
        
    #     alpha = anum * np.pi/180.
    #     beta  = bnum * np.pi/180.
    #     gamma = gnum * np.pi/180.
        
    #     A[0,0] = np.cos(beta) * np.cos(gamma)
    #     A[0,1] = np.sin(alpha) * np.sin(beta) * np.cos(gamma) - np.cos(alpha) * np.sin(gamma)
    #     A[0,2] = np.cos(alpha) * np.sin(beta) * np.cos(gamma) + np.sin(alpha) * np.sin(gamma)
    #     A[1,0] = np.cos(beta) * np.sin(gamma)
    #     A[1,1] = np.sin(alpha) * np.sin(beta) * np.sin(gamma) + np.cos(alpha) * np.cos(gamma)
    #     A[1,2] = np.cos(alpha) * np.sin(beta) * np.sin(gamma) - np.sin(alpha) * np.cos(gamma)
    #     A[2,0] = - np.sin(beta)
    #     A[2,1] = np.sin(alpha) * np.cos(beta)
    #     A[2,2] = np.cos(alpha) * np.cos(beta)
        
    #     grid = np.zeros([3,NUM,NUM,NUM]) 
    #     for i in range(NUM):
    #         for j in range(NUM):
    #             for k in range(NUM):
    #                 grid[:,i,j,k] = matmul( A, gridn[:,i,j,k] )
    # else:
    #     grid = gridn

    # for i in range(NUM):
    #     for j in range(NUM):
    #         for k in range(NUM):
    #             # B = [y, x, z], with curlB = [-1,-1,-1]
    #             B[0,i,j,k] = grid[1,i,j,k]
    #             B[1,i,j,k] = grid[2,i,j,k]
    #             B[2,i,j,k] = grid[0,i,j,k]
    
    # x_ = grid[0,:,:,:]
    # y_ = grid[1,:,:,:]
    # z_ = grid[2,:,:,:]
    
    # measure_ = measure #[0:nI,0:nJ,0:nK]

    # bx_ = B[0,0:nI,0:nJ,0:nK]
    # by_ = B[1,0:nI,0:nJ,0:nK]
    # bz_ = B[2,0:nI,0:nJ,0:nK]

    # current = jcalc2ptr.jcalc2ptr( bx_, by_, bz_, x_, y_, z_, measure_, nI, nJ, nK, 
    #           nI+1, nJ+1, nK+1, nJ+2 )
    
    # current = current / (7.9577471 * 10**3)
    
    # cx = current[0,:,1:-1,0:-1]
    # cy = current[1,:,1:-1,0:-1]
    # cz = current[2,:,1:-1,0:-1]

    # gridx = grid[0,:,:,:]
    # gridx = gridx.T
    # gridx = gridx.reshape(-1)
    
    # gridy = grid[1,:,:,:]
    # gridy = gridy.T
    # gridy = gridy.reshape(-1)
    
    # gridz = grid[2,:,:,:]
    # gridz = gridz.T
    # gridz = gridz.reshape(-1)
    
    # measure = measure.T
    # measure = measure.reshape(-1)
    
    # bx = B[0,:,:,:]
    # bx = bx.T
    # bx = bx.reshape(-1)
    
    # by = B[1,:,:,:]
    # by = by.T
    # by = by.reshape(-1)
    
    # bz = B[2,:,:,:]
    # bz = bz.T
    # bz = bz.reshape(-1)
    
    # jx,jy,jz = get_lfm_current_sub2( gridx, gridy, gridz, bx, by, bz, nI, nJ, nK )
    
    # jx = jx / (7.9577471 * 10**9)
    # jy = jy / (7.9577471 * 10**9)
    # jz = jz / (7.9577471 * 10**9)
    
    # print('Done comparing current density')

    # # illustrate structure of lfm grid
    # import matplotlib.pyplot as plt

    # xcellcenter_ = lfmdata.xsliceSM
    # rcellcenter_ = lfmdata.rsliceSM
    # acellcenter_ = lfmdata.asliceSM
    
    # # slices in azimuth are identical when looking at cylinderical coordinates

    # # azimuth range
    # plt.plot(acellcenter_[:]*180/np.pi, '.')
    # plt.ylabel( 'Azimuth (degree)')
    # plt.xlabel( 'Slice number')
    # plt.title('Azimuth wraps on last slice')
    # plt.show()

    # # full x,r grid on one azimuth slice
    # for i in range(xcellcenter_.shape[0]):
    #     plt.plot(xcellcenter_[i,:], rcellcenter_[i,:], '.')
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'r (Re)')
    # plt.title('Full grid on one azimuth slice')
    # plt.show()

    # # one distorted semicircle on one azimuth slice
    # plt.plot(xcellcenter_[100,:], rcellcenter_[100,:],'.')
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'r (Re)')
    # plt.title('Single distorted semicircle')
    # plt.show()

    # # outer boundary.  Note, if you include cell vertices,
    # # it is a cylinder
    # plt.plot(xcellcenter_[-1,:], rcellcenter_[-1,:],'.')
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'r (Re)')
    # plt.title('Outer set of points')
    # plt.show()

    # LAYER = 50
    # xmesh  = lfmdata.DataArray[lfmdata.varidx['x'],:,LAYER,:]
    # ymesh  = lfmdata.DataArray[lfmdata.varidx['y'],:,LAYER,:]
    # zmesh  = lfmdata.DataArray[lfmdata.varidx['z'],:,LAYER,:]
    # # jxmesh = lfmdata.DataArray[lfmdata.varidx['jx'],:,LAYER,:]
    # # jymesh = lfmdata.DataArray[lfmdata.varidx['jy'],:,LAYER,:]
    # # jzmesh = lfmdata.DataArray[lfmdata.varidx['jz'],:,LAYER,:]
    # # jxmesh = lfmdata.DataArray[lfmdata.varidx['jx'],LAYER,:,:]
    # # jymesh = lfmdata.DataArray[lfmdata.varidx['jy'],LAYER,:,:]
    # # jzmesh = lfmdata.DataArray[lfmdata.varidx['jz'],LAYER,:,:]
    # # jxmesh = lfmdata.DataArray[lfmdata.varidx['jx'],:,:,LAYER]
    # # jymesh = lfmdata.DataArray[lfmdata.varidx['jy'],:,:,LAYER]
    # # jzmesh = lfmdata.DataArray[lfmdata.varidx['jz'],:,:,LAYER]
    # # bxmesh = lfmdata.DataArray[lfmdata.varidx['bx'],:,LAYER,:]
    # # bymesh = lfmdata.DataArray[lfmdata.varidx['by'],:,LAYER,:]
    # # bzmesh = lfmdata.DataArray[lfmdata.varidx['bz'],:,LAYER,:]
    # jxmesh = lfmdata.jx[:,LAYER,:]
    # jymesh = lfmdata.jy[:,LAYER,:]
    # jzmesh = lfmdata.jz[:,LAYER,:]
    # jx2mesh = lfmdata.jx2[:,LAYER,:]
    # jy2mesh = lfmdata.jy2[:,LAYER,:]
    # jz2mesh = lfmdata.jz2[:,LAYER,:]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_wireframe(xmesh, zmesh, jxmesh, label='jx')
    # ax.set_xlabel( r'$x$' )
    # ax.set_ylabel( r'$z$' )
    # ax.set_zlabel( r'$j_z$' )

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_wireframe(xmesh, zmesh, jymesh, label='jy')
    # ax.set_xlabel( r'$x$' )
    # ax.set_ylabel( r'$z$' )
    # ax.set_zlabel( r'$j_z$' )

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot_wireframe(xmesh, zmesh, jzmesh, label='jz')
    # ax.set_xlabel( r'$x$' )
    # ax.set_ylabel( r'$z$' )
    # ax.set_zlabel( r'$j_z$' )

    # # full x,r grid on one azimuth slice
    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jxmesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jx (Re)')
    # plt.title('jx on one azimuth slice')
    # plt.show()

    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jymesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jy (Re)')
    # plt.title('jy on one azimuth slice')
    # plt.show()

    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jzmesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jz (Re)')
    # plt.title('jz on one azimuth slice')
    # plt.show()

    # # full x,r grid on one azimuth slice
    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jx2mesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jx2 (Re)')
    # plt.title('jx2 on one azimuth slice')
    # plt.show()

    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jy2mesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jy2 (Re)')
    # plt.title('jy2 on one azimuth slice')
    # plt.show()

    # for i in range(xmesh.shape[0]):
    #     plt.plot(xmesh[i,:], np.log(np.abs(jz2mesh[i,:])))
    # plt.xlabel( 'x (Re)')
    # plt.ylabel( 'jz2 (Re)')
    # plt.title('jz2 on one azimuth slice')
    # plt.show()


    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.plot_wireframe(xmesh, ymesh, zmesh, label='xyz')
    # # ax.set_xlabel( r'$x$' )
    # # ax.set_ylabel( r'$z$' )
    # # ax.set_zlabel( r'$z$' )
    
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.plot_wireframe(xmesh, zmesh, jxmesh, label='jx')
    # # ax.set_xlabel( r'$x$' )
    # # ax.set_ylabel( r'$z$' )
    # # ax.set_zlabel( r'$j_x$' )

    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.plot_wireframe(xmesh, zmesh, jymesh, label='jy')
    # # ax.set_xlabel( r'$x$' )
    # # ax.set_ylabel( r'$z$' )
    # # ax.set_zlabel( r'$j_y$' )

    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # ax.plot_wireframe(xmesh, zmesh, jzmesh, label='jz')
    # # ax.set_xlabel( r'$x$' )
    # # ax.set_ylabel( r'$z$' )
    # # ax.set_zlabel( r'$j_z$' )




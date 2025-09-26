#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:38:31 2024

@author: Dean Thomas
"""

from numba import njit
import numpy as np

@njit
def calcCurlB(DA, i, j, k, n, nI, nJ, nK, nBlock, _x, _y, _z, _bx, _by, _bz):
    """ Subroutine that allows numba accelleration. It calculates curl 
    of B at point i,j,k in block n using data from BATSRUS file 
        
    Inputs:
        DA = BATSRUS DataArray
        
        i,j,k,n = grid coordinates of point i,j,k inside block n
        
        nI,nJ,nK = number of x,y,z points, respectively, in block n. Provided to 
            avoid constantly looking them up
            
        nBlock = total number of blocks
        
        _x,_y,_z = batsrus.varidx values for x, y, z.  Provided to avoid 
            constantly looking them up
        
        _bx,_by,_bz = batsrus.varidx values for bx, by, bz.  Provided to avoid 
            constantly looking them up
                       
    Outputs:
        curlB = curl of B at point i,j,k (in GSM coordinates)
    """
    
    assert( i>=0 and i<nI )
    assert( j>=0 and j<nJ )
    assert( k>=0 and k<nK )
    assert( n>=0 and n<nBlock )
    
    # Use 2nd order stencils to calculate derivatives, sum derivatives to 
    # determine curlB

    dX = DA[_x,1,0,0,n] - DA[_x,0,0,0,n] 
    dY = DA[_y,0,1,0,n] - DA[_y,0,0,0,n] 
    dZ = DA[_z,0,0,1,n] - DA[_z,0,0,0,n] 

    # derivatives over x

    if i > 0 and i < nI-1: # in interior of block n
        curlByx = (DA[_by, i+1, j, k, n] - DA[_by, i-1, j, k, n])/(2*dX)
    elif i == 0: # on face
        curlByx = (-3*DA[_by, 0, j, k, n] + 4*DA[_by, 1, j, k, n]
                - DA[_by, 2, j, k, n])/(2*dX)
    else: # i == nI-1: on face
        curlByx = (3*DA[_by, nI-1, j, k, n] - 4*DA[_by, nI-2, j, k, n]
                + DA[_by, nI-3, j, k, n])/(2*dX)        
    
    if i > 0 and i < nI-1: # in interior of block n
        curlBzx = (DA[_bz, i+1, j, k, n] - DA[_bz, i-1, j, k, n])/(2*dX)
    elif i == 0: # on face
        curlBzx = (-3*DA[_bz, 0, j, k, n] + 4*DA[_bz, 1, j, k, n]
                - DA[_bz, 2, j, k, n])/(2*dX)
    else: # i == nI-1: on face
        curlBzx = (3*DA[_bz, nI-1, j, k, n] - 4*DA[_bz, nI-2, j, k, n]
                + DA[_bz, nI-3, j, k, n])/(2*dX)        
    
    # derivatives over y

    if j > 0 and j < nJ-1: # in interior of block n
        curlBxy = (DA[_bx, i, j+1, k, n] - DA[_bx, i, j-1, k, n])/(2*dY)
    elif j == 0: # on face
        curlBxy = (-3*DA[_bx, i, 0, k, n] + 4*DA[_bx, i, 1, k, n]
                - DA[_bx, i, 2, k, n])/(2*dY)
    else: # j == nJ-1: on face
        curlBxy = (3*DA[_bx, i, nJ-1, k, n] - 4*DA[_bx, i, nJ-2, k, n]
                + DA[_bx, i, nJ-3, k, n])/(2*dY)        
    
    if j > 0 and j < nJ-1: # in interior of block n
        curlBzy = (DA[_bz, i, j+1, k, n] - DA[_bz, i, j-1, k, n])/(2*dY)
    elif j == 0: # on face
        curlBzy = (-3*DA[_bz, i, 0, k, n] + 4*DA[_bz, i, 1, k, n]
                - DA[_bz, i, 2, k, n])/(2*dY)
    else: # j == nJ-1: on face
        curlBzy = (3*DA[_bz, i, nJ-1, k, n] - 4*DA[_bz, i, nJ-2, k, n]
                + DA[_bz, i, nJ-3, k, n])/(2*dY)        
    
    # derivatives over z

    if k > 0 and k < nK-1: # in interior of block n
        curlBxz = (DA[_bx, i, j, k+1, n] - DA[_bx, i, j, k-1, n])/(2*dZ)
    elif k == 0: # on face
        curlBxz = (-3*DA[_bx, i, j, 0, n] + 4*DA[_bx, i, j, 1, n]
                - DA[_bx, i, j, 2, n])/(2*dZ)
    else: # k == nK-1: on face
        curlBxz = (3*DA[_bx, i, j, nK-1, n] - 4*DA[_bx, i, j, nK-2, n]
                + DA[_bx, i, j, nK-3, n])/(2*dZ)        
    
    if k > 0 and k < nK-1: # in interior of block n
        curlByz = (DA[_by, i, j, k+1, n] - DA[_by, i, j, k-1, n])/(2*dZ)
    elif k == 0: # on face
        curlByz = (-3*DA[_by, i, j, 0, n] + 4*DA[_by, i, j, 1, n]
                - DA[_by, i, j, 2, n])/(2*dZ)
    else: # k == nK-1: on face
        curlByz = (3*DA[_by, i, j, nK-1, n] - 4*DA[_by, i, j, nK-2, n]
                + DA[_by, i, j, nK-3, n])/(2*dZ)        
    
    # Combine derivatives to get curl
    return curlBzy-curlByz, curlBxz-curlBzx, curlByx-curlBxy

@njit
def BATSRUS_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, nBlock, rCurrents):
    """ Subroutine that allows numba accelleration.  It calculates current density
    j from the curl of B field using data from a BATSRUS file.
    
    Inputs:        
        data_arr: numpy array in which the bastrus data is stored
        
        DataArray: reshaped data_arr, provided to allow numba acceleration
        
        varidx: variable indices in data_arr, e.g., x data is at varidx['x'] 
        
        nVar: how many variables in data_arr

        nI, nJ, nK: number of points along x,y,z axes in cartesian grid
        
    Outputs:
        data_arr = updated data_arr with calculated values of jx, jy, jz 
    """

    # We use these values throughout the routine, so to avoid muliple lookups
    # we look for them once.
    _x = varidx['x']
    _y = varidx['y']
    _z = varidx['z']
    
    _bx = varidx['bx']
    _by = varidx['by']
    _bz = varidx['bz']
    
    _jx = varidx['jx']
    _jy = varidx['jy']
    _jz = varidx['jz']
    
    # Iterate thru points in simulation grid, calculating the current density
    # based on curl of B at each point.  
    for n in range(nBlock):
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):
                                   
                    # Distance from center of earth to point i,j,k
                    r0 = np.sqrt(DataArray[_x,i,j,k,n]**2 +
                                 DataArray[_y,i,j,k,n]**2 +
                                 DataArray[_z,i,j,k,n]**2)
                    
                    # Only include point if it is outside of rCurrents
                    # Data are not valid inside rCurrents
                    if r0 >= rCurrents:
                        
                        ##########################################################
                        # Below we calculate j in each differential volume 
                        # element in the integral.  We want the final result to be 
                        # in microamps/m^2.
                        #
                        # j = 1/mu0 curl B
                        #
                        # where mu0 = 4pi 10^-7 [kg m/s^2/A^2]
                        #       curl B = [nT]/[Re]
                        #              = 10^-9 [kg/s^2/A] / 6.371 10^6 [m]
                        #
                        # j = 10^7/4pi [A^2 s^2/ kg / m] 10^-9 [kg/s^2/A] / 6.371 10^6 [m]
                        #   = 1/(4pi) 10^-8  1/6.371 A/m^2  with 10^6 microAmp/A
                        #   = 1/(4pi) 1/6.371 10^-2 microAmp/m2 
                        #   = 1.2491 10^-4 [microAmp/m^2] with distances in Re, B in nT
                        ##########################################################
        
                        curlBx, curlBy, curlBz = calcCurlB(DataArray, i, j, k, n, nI, nJ, nK, nBlock, _x, _y, _z, _bx, _by, _bz)
                        DataArray[_jx,i,j,k,n] = curlBx * 1.2491 * 10**(-4) 
                        DataArray[_jy,i,j,k,n] = curlBy * 1.2491 * 10**(-4)
                        DataArray[_jz,i,j,k,n] = curlBz * 1.2491 * 10**(-4)
                        
                    else:
                        DataArray[_jx,i,j,k,n] = 0. 
                        DataArray[_jy,i,j,k,n] = 0.
                        DataArray[_jz,i,j,k,n] = 0.
                        
    return data_arr

if __name__ == "__main__":
        
    # Test curl B operations against known B fields

    import os.path
         
    data_dir = r'/Volumes/PhysicsHD'
    # data_dir = r'/Volumes/Data1'
    
    info = {
            "model": "SWMF",
            "run_name": "Bob_Weigel_070323_3",
            # "rCurrents": 3.0,
            "rIonosphere": 1.01725,
            "file_type": "cdf",
            "method": "method1",
            "dir_run": os.path.join(data_dir, "Bob_Weigel_070323_3"),
            "dir_plots": os.path.join(data_dir, "Bob_Weigel_070323_3.plots"),
            "dir_derived": os.path.join(data_dir, "Bob_Weigel_070323_3.derived"),
            "dir_magnetosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "GM_CDF"),
            "dir_ionosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "IONO-2D_CDF")
    }
    
    file = '/Volumes/PhysicsHD/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-194800-000.out.cdf'
    # file = '/Volumes/Data1/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-194800-000.out.cdf'
    
    from deltaB import get_batsrus_data_from_cdf
    
    batsrus = get_batsrus_data_from_cdf(file, info)
    
    nI        = batsrus.nI
    nJ        = batsrus.nJ
    nK        = batsrus.nK
    nBlock    = batsrus.nBlock
    nVar      = batsrus.DataArray.shape[0]
    
    DataArray = batsrus.DataArray
    data_arr  = batsrus.data_arr
    varidx    = batsrus.varidx
    rCurrents = batsrus.rCurrents
    
    _x = batsrus.varidx['x']
    _y = batsrus.varidx['y']
    _z = batsrus.varidx['z']
    
    _bx = batsrus.varidx['bx']
    _by = batsrus.varidx['by']
    _bz = batsrus.varidx['bz']
    
    _jx = batsrus.varidx['jx']
    _jy = batsrus.varidx['jy']
    _jz = batsrus.varidx['jz']
    
    # # This should give us curlB=(1,1,1)
    # data_arr[:,_bx] = data_arr[:,_z]
    # data_arr[:,_by] = data_arr[:,_x]
    # data_arr[:,_bz] = data_arr[:,_y]
    # value = np.array([1.,1.,1.])
    # value2 = 3. * (1.2491 * 10**(-4))**2
    
    # This should give us curlB=(1000,10,100)
    data_arr[:,_bx] = 10.*data_arr[:,_z]
    data_arr[:,_by] = 100.*data_arr[:,_x]
    data_arr[:,_bz] = 1000.*data_arr[:,_y]
    value = np.array([1000.,10.,100.])
    value2 = (1000.*1000. + 10.*10. + 100.*100.) * (1.2491 * 10**(-4))**2
    
    # # This should give us curlB=(0,0,0)
    # data_arr[:,_bx] = data_arr[:,_x]
    # data_arr[:,_by] = data_arr[:,_y]
    # data_arr[:,_bz] = data_arr[:,_z]
    # value = np.array([0.0,0.0,0.0])
    # value2 = 0. * (1.2491 * 10**(-4))**2
    
    if True: # test curl function
    
        # Calculate curlB for each point on grid
        # Verify that we get the expected answer
        for n in range(nBlock):
            for i in range(nI):
                for j in range(nJ):
                    for k in range(nK):
                        curlB = calcCurlB(DataArray, i, j, k, n, nI, nJ, nK, nBlock, _x, _y, _z, _bx, _by, _bz )
                        if np.linalg.norm(curlB - value) > 0.000000001: print( i,j,k,curlB )
        print('If no values shown, test passed')
        print('Done Curl')
        
    if True: # test curl B to J
       
        data_arr = BATSRUS_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, nBlock, rCurrents)
        
        # Reshape the data array     
        DataArray = data_arr.transpose()
        assert(np.isfortran(DataArray))
        
        DataArray = DataArray.reshape((nVar, nI, nJ, nK,nBlock), order='F')
        assert(np.isfortran(DataArray))

        for n in range(nBlock):
            for i in range(nI):
                for j in range(nJ):
                    for k in range(nK):
                        jx = DataArray[_jx,i,j,k,n]
                        jy = DataArray[_jy,i,j,k,n]
                        jz = DataArray[_jz,i,j,k,n]
                        j2 = jx*jx + jy*jy + jz*jz
                        
                        r0 = np.sqrt(DataArray[_x,i,j,k,n]**2 +
                                     DataArray[_y,i,j,k,n]**2 +
                                     DataArray[_z,i,j,k,n]**2)
                        if r0 > rCurrents:
                            if abs(j2 - value2) > 0.000000001: print( i,j,k,n,j2 )
        print('If no values shown, test passed')
        print('Done j')

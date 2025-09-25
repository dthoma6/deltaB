#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:38:31 2024

@author: Dean Thomas
"""

from numba import njit
import numpy as np

@njit
def calcCurlB(DA, i, j, k, nI, nJ, nK, _x, _y, _z, _bx, _by, _bz):
    """ Subroutine that allows numba accelleration. It calculates curl 
    of B at point i,j,k using data from openggcm file 
    
    Inputs:
        DA = openggcm DataArray
        
        i,j,k = grid coordinates of point
        
        nI,nJ,nK = number of x,y,z points in grid. Provided to avoid constantly 
            looking them up
        
        _x,_y,_z = openggcm.varidx values for x, y, z.  Provided to avoid 
            constantly looking them up
                       
        _bx,_by,_bz = openggcm.varidx values for bx, by, bz.  Provided to avoid 
            constantly looking them up
                       
    Outputs:
        curlB = curl of B at point i,j,k (in GSM coordinates)
    """
    
    assert( i>=0 and i<nI )
    assert( j>=0 and j<nJ )
    assert( k>=0 and k<nK )
        
    # Use stencils to calculate derivatives, sum derivatives to determine curlB
    # We have unequal intervals, so we must use the correct stencils
    #
    # Singh, Ashok K., and B. S. Bhadauria. "Finite difference formulae for 
    # unequal sub-intervals using Lagrange’s interpolation formula." Int. J. 
    # Math. Anal 3.17 (2009): 815.
    
    # derivatives over x
    
    if i > 0 and i < nI-1: # in interior
        h1 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        h2 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        f0 = DA[_by,i-1,j,k]
        f1 = DA[_by,i  ,j,k]
        f2 = DA[_by,i+1,j,k]
        curlByx = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif i == 0: # on face
        h1 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        h2 = DA[_x ,i+2,j,k] - DA[_x,i+1,j,k]
        f0 = DA[_by,i  ,j,k]
        f1 = DA[_by,i+1,j,k]
        f2 = DA[_by,i+2,j,k]
        curlByx = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # i == nI-1: on face
        h1 = DA[_x ,i-1,j,k] - DA[_x,i-2,j,k]
        h2 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        f0 = DA[_by,i-2,j,k]
        f1 = DA[_by,i-1,j,k]
        f2 = DA[_by,i  ,j,k]
        curlByx =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    if i > 0 and i < nI-1: # in interior
        h1 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        h2 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        f0 = DA[_bz,i-1,j,k]
        f1 = DA[_bz,i  ,j,k]
        f2 = DA[_bz,i+1,j,k]
        curlBzx = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif i == 0: # on face
        h1 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        h2 = DA[_x ,i+2,j,k] - DA[_x,i+1,j,k]
        f0 = DA[_bz,i  ,j,k]
        f1 = DA[_bz,i+1,j,k]
        f2 = DA[_bz,i+2,j,k]
        curlBzx = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # i == nI-1: on face
        h1 = DA[_x ,i-1,j,k] - DA[_x,i-2,j,k]
        h2 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        f0 = DA[_bz,i-2,j,k]
        f1 = DA[_bz,i-1,j,k]
        f2 = DA[_bz,i  ,j,k]
        curlBzx =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   

    # derivatives over y
                    
    if j > 0 and j < nJ-1: # in interior
        h1 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        h2 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        f0 = DA[_bx,i,j-1,k]
        f1 = DA[_bx,i,j  ,k]
        f2 = DA[_bx,i,j+1,k]
        curlBxy = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif j == 0: # on face
        h1 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        h2 = DA[_y ,i,j+2,k] - DA[_y,i,j+1,k]
        f0 = DA[_bx,i,j  ,k]
        f1 = DA[_bx,i,j+1,k]
        f2 = DA[_bx,i,j+2,k]
        curlBxy = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # j == nJ-1: on face
        h1 = DA[_y ,i,j-1,k] - DA[_y,i,j-2,k]
        h2 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        f0 = DA[_bx,i,j-2,k]
        f1 = DA[_bx,i,j-1,k]
        f2 = DA[_bx,i,j  ,k]  
        curlBxy =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    if j > 0 and j < nJ-1: # in interior
        h1 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        h2 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        f0 = DA[_bz,i,j-1,k]
        f1 = DA[_bz,i,j  ,k]
        f2 = DA[_bz,i,j+1,k]
        curlBzy = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif j == 0: # on face
        h1 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        h2 = DA[_y ,i,j+2,k] - DA[_y,i,j+1,k]
        f0 = DA[_bz,i,j  ,k]
        f1 = DA[_bz,i,j+1,k]
        f2 = DA[_bz,i,j+2,k]
        curlBzy = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # j == nJ-1: on face
        h1 = DA[_y ,i,j-1,k] - DA[_y,i,j-2,k]
        h2 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        f0 = DA[_bz,i,j-2,k]
        f1 = DA[_bz,i,j-1,k]
        f2 = DA[_bz,i,j  ,k]  
        curlBzy =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    # derivatives over z
    
    if k > 0 and k < nK-1: # in interior
        h1 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        h2 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        f0 = DA[_bx,i,j,k-1]
        f1 = DA[_bx,i,j,k  ]
        f2 = DA[_bx,i,j,k+1]
        curlBxz = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif k == 0: # on face
        h1 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        h2 = DA[_z ,i,j,k+2] - DA[_z,i,j,k+1]
        f0 = DA[_bx,i,j,k  ]
        f1 = DA[_bx,i,j,k+1]
        f2 = DA[_bx,i,j,k+2]
        curlBxz = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # k == nK-1: on face
        h1 = DA[_z ,i,j,k-1] - DA[_z,i,j,k-2]
        h2 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        f0 = DA[_bx,i,j,k-2]
        f1 = DA[_bx,i,j,k-1]
        f2 = DA[_bx,i,j,k  ]
        curlBxz =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    if k > 0 and k < nK-1: # in interior
        h1 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        h2 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        f0 = DA[_by,i,j,k-1]
        f1 = DA[_by,i,j,k  ]
        f2 = DA[_by,i,j,k+1]
        curlByz = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif k == 0: # on face
        h1 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        h2 = DA[_z ,i,j,k+2] - DA[_z,i,j,k+1]
        f0 = DA[_by,i,j,k  ]
        f1 = DA[_by,i,j,k+1]
        f2 = DA[_by,i,j,k+2]
        curlByz = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # k == nK-1: on face
        h1 = DA[_z ,i,j,k-1] - DA[_z,i,j,k-2]
        h2 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        f0 = DA[_by,i,j,k-2]
        f1 = DA[_by,i,j,k-1]
        f2 = DA[_by,i,j,k  ]
        curlByz =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                
    # Combine derivatives to get curl
    
    return curlBzy-curlByz, curlBxz-curlBzx, curlByx-curlBxy

@njit
def OpenGGCM_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, rCurrents):
    """ Subroutine that allows numba accelleration.  It calculates current density
    j from the curl of B field using data from a OpenGGCM file.
    
    Inputs:        
        data_arr: numpy array in which the openggcm data is stored
        
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
    for i in range(nI):
        for j in range(nJ):
            for k in range(nK):
                               
                # Distance from center of earth to point i,j,k
                r0 = np.sqrt(DataArray[_x,i,j,k]**2 +
                             DataArray[_y,i,j,k]**2 +
                             DataArray[_z,i,j,k]**2)
                
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
    
                    curlBx, curlBy, curlBz = calcCurlB(DataArray, i, j, k, nI, nJ, nK, _x, _y, _z, _bx, _by, _bz)
                    DataArray[_jx,i,j,k] = curlBx * 1.2491 * 10**(-4) 
                    DataArray[_jy,i,j,k] = curlBy * 1.2491 * 10**(-4)
                    DataArray[_jz,i,j,k] = curlBz * 1.2491 * 10**(-4)
                    
                else:
                    DataArray[_jx,i,j,k] = 0. 
                    DataArray[_jy,i,j,k] = 0.
                    DataArray[_jz,i,j,k] = 0.
                        
    return data_arr

if __name__ == "__main__":

 
    # Test curl B operations against known B fields

    import os.path

    # Read in a OpenGGCM file
    
    data_dir = r'/Volumes/PhysicsHD'
    # data_dir = r'/Volumes/Data2'
    
    info = {
            "model": "OpenGGCM",
            "run_name": "Dean_Thomas_052924_1",
            # "rCurrents": 3.0,
            "rIonosphere": 1.01725,
            "file_type": "cdf",
            "method": "method1",
            "dir_run": os.path.join(data_dir, "Dean_Thomas_052924_1"),
            "dir_plots": os.path.join(data_dir, "Dean_Thomas_052924_1.plots"),
            "dir_derived": os.path.join(data_dir, "Dean_Thomas_052924_1.derived"),
            "dir_magnetosphere": os.path.join(data_dir, "Dean_Thomas_052924_1", "GM_CDF"),
            "dir_ionosphere": os.path.join(data_dir, "Dean_Thomas_052924_1", "IONO-2D_CDF")
            }

    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.023400.cdf'
    # file = '/Volumes/Data2/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.023400.cdf'
    
    from deltaB import get_openggcm_data_from_cdf

    oggcm = get_openggcm_data_from_cdf(file, info)
    
    DataArray = oggcm.DataArray
    data_arr  = oggcm.data_arr
    varidx    = oggcm.varidx
    rCurrents = oggcm.rCurrents
    
    nVar = len(varidx)
    
    _x = oggcm.varidx['x']
    _y = oggcm.varidx['y']
    _z = oggcm.varidx['z']

    _bx = oggcm.varidx['bx']
    _by = oggcm.varidx['by']
    _bz = oggcm.varidx['bz']
    
    _jx = oggcm.varidx['jx']
    _jy = oggcm.varidx['jy']
    _jz = oggcm.varidx['jz']

    nI  = oggcm.nI
    nJ  = oggcm.nJ
    nK  = oggcm.nK

    x = data_arr[:,_x]
    y = data_arr[:,_y]
    z = data_arr[:,_z]
          
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
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):
                    curlB = calcCurlB(DataArray, i, j, k, nI, nJ, nK, _x, _y, _z, _bx, _by, _bz )
                    if np.linalg.norm(curlB - value) > 0.000000001: print( i,j,k,curlB )
        print('If no values shown, test passed')
        print('Done Curl')
        
    if True: # test curl B to J
       
        data_arr = OpenGGCM_curlBtoJ(data_arr, DataArray, varidx, nVar, nI, nJ, nK, rCurrents)
        
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):
                    jx = DataArray[_jx,i,j,k]
                    jy = DataArray[_jy,i,j,k]
                    jz = DataArray[_jz,i,j,k]
                    j2 = jx*jx + jy*jy + jz*jz
                    
                    r0 = np.sqrt(DataArray[_x,i,j,k]**2 +
                                 DataArray[_y,i,j,k]**2 +
                                 DataArray[_z,i,j,k]**2)
                    if r0 > rCurrents:
                        if abs(j2 - value2) > 0.000000001: print( i,j,k,j2 )
        print('If no values shown, test passed')
        print('Done j')



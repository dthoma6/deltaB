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
        
    # Use stencils to calculate derivatives, sum derivatives to determine divB
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
    
    # Reshape the data array     
    # DataArray = data_arr.transpose()
    # assert(np.isfortran(DataArray))
    
    # DataArray = DataArray.reshape((nVar, nI, nJ, nK), order='F')
    # assert(np.isfortran(DataArray))
    
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

        from OpenGGCM_dataframe import get_openggcm_grid_sub
        
        nI = 10
        nJ = 10
        nK = 10
        
        x_ = -np.array([9,8,7,6,5,4,3,2,1,0])
        y_ = -np.array([9,8,7,6,5,4,3,2,1,0])        
        z_ =  np.array([0,1,2,3,4,5,6,7,8,9])
        
        x_ = -x_**2
        y_ = -y_**2
        z_ =  z_**2
        
        # Nonsense cell vertex data
        # Used to determine measure, which I don't need for this test
        xcell_ =  (x_ - 0.5)
        ycell_ =  (y_ - 0.5)
        zcell_ =  (z_ - 0.5)
        
        # Setup grid
        x, y, z, measure = get_openggcm_grid_sub( x_, y_, z_, 
                                                    xcell_, ycell_, zcell_,
                                                    nI, nJ, nK )
         
        # This should give us curlB=(1,1,1)
        bx = z
        by = x
        bz = y
        value = np.array([1.,1.,1.])
        value2 = 3.
        
        # # This should give us curlB=(-1000,-10,-100)
        # bx = 100.*y
        # by = 1000.*z
        # bz = 10.*x
        # value = np.array([-1000.,-10.,-100.])
        # value2 = 1000.*1000. + 10.*10. + 100.*100.
        
        # # This should give us curlB=(0,0,0)
        # bx = x
        # by = y
        # bz = z
        # value = np.array([0.0,0.0,0.0])
        # value2 = 0.
        
        if False: # test curl function
            # Create data array
            data_arr = np.zeros((len(x),6))
            data_arr[:,0] = x
            data_arr[:,1] = y
            data_arr[:,2] = z
            data_arr[:,3] = bx
            data_arr[:,4] = by
            data_arr[:,5] = bz
            
            DataArray = data_arr.transpose()
            DataArray = DataArray.reshape((6, nI, nJ, nK), order='F')
    
            # Calculate curlB for each point on grid
            # Verify that we get the expected answer
            for i in range(nI):
                for j in range(nJ):
                    for k in range(nK):
                        curlB = calcCurlB(DataArray, i, j, k, nI, nJ, nK, 0,1,2,3,4,5 )
                        if np.linalg.norm(curlB - value) > 0.000000001: print( i,j,k,curlB )
            print('Done')
            
        if True: # test curl B to J
            # Create data array
            data_arr = np.zeros((len(x),9))
            data_arr[:,0] = x
            data_arr[:,1] = y
            data_arr[:,2] = z
            data_arr[:,3] = bx
            data_arr[:,4] = by
            data_arr[:,5] = bz
            data_arr[:,6] = 0
            data_arr[:,7] = 0
            data_arr[:,8] = 0
            
            import numba
            
            varidx = numba.typed.Dict.empty(key_type=numba.types.unicode_type, 
                                            value_type=numba.types.int64,)
            
            varidx['x']  = 0
            varidx['y']  = 1
            varidx['z']  = 2
            varidx['bx'] = 3
            varidx['by'] = 4
            varidx['bz'] = 5
            varidx['jx'] = 6
            varidx['jy'] = 7
            varidx['jz'] = 8
            nVar = len(varidx)
            
            # DataArray = data_arr.transpose()
            # DataArray = DataArray.reshape((9, nI, nJ, nK), order='F')
            
            data_arr = OpenGGCM_curlBtoJ(data_arr, varidx, nVar, nI, nJ, nK)
            
            jx = data_arr[:,6]
            jy = data_arr[:,7]
            jz = data_arr[:,8]
            j2 = jx*jx + jy*jy + jz*jz
            
            for i in range(len(j2)):
                if value2 > 0.:
                    if abs(j2[i] - value2)/value2 > 0.000000001: print( i,j2[i] )
                else:
                    if abs(j2[i] - value2)       > 0.000000001: print( i,j2[i] )
            print('Done')

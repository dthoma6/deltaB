#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:38:31 2024

@author: Dean Thomas
"""

from numba import njit
import numpy as np

@njit
def calcDivB(DA, i, j, k, nI, nJ, nK, _x, _y, _z, _bx, _by, _bz):
    """ Subroutine for calc_ms_divBint_b_sub that allows numba accelleration. It  
    calculates divergence of B at point i,j,k using data from openggcm file 
    and the Helmholtz decompostion theorem to replace Biot-Savart with a volume 
    integral over the divergence of B.
    
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
        divB = divergence of B at point i,j,k, which is sum of 
            divBx, divBy and divBz (in GSM coordinates)
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
    
    if i > 0 and i < nI-1: # in interior
        h1 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        h2 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        f0 = DA[_bx,i-1,j,k]
        f1 = DA[_bx,i  ,j,k]
        f2 = DA[_bx,i+1,j,k]
        divBx = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif i == 0: # on face
        h1 = DA[_x ,i+1,j,k] - DA[_x,i  ,j,k]
        h2 = DA[_x ,i+2,j,k] - DA[_x,i+1,j,k]
        f0 = DA[_bx,i  ,j,k]
        f1 = DA[_bx,i+1,j,k]
        f2 = DA[_bx,i+2,j,k]
        divBx = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # i == nI-1: on face
        h1 = DA[_x ,i-1,j,k] - DA[_x,i-2,j,k]
        h2 = DA[_x ,i  ,j,k] - DA[_x,i-1,j,k]
        f0 = DA[_bx,i-2,j,k]
        f1 = DA[_bx,i-1,j,k]
        f2 = DA[_bx,i  ,j,k]
        divBx =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    if j > 0 and j < nJ-1: # in interior
        h1 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        h2 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        f0 = DA[_by,i,j-1,k]
        f1 = DA[_by,i,j  ,k]
        f2 = DA[_by,i,j+1,k]
        divBy = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif j == 0: # on face
        h1 = DA[_y ,i,j+1,k] - DA[_y,i,j  ,k]
        h2 = DA[_y ,i,j+2,k] - DA[_y,i,j+1,k]
        f0 = DA[_by,i,j  ,k]
        f1 = DA[_by,i,j+1,k]
        f2 = DA[_by,i,j+2,k]
        divBy = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # j == nJ-1: on face
        h1 = DA[_y ,i,j-1,k] - DA[_y,i,j-2,k]
        h2 = DA[_y ,i,j  ,k] - DA[_y,i,j-1,k]
        f0 = DA[_by,i,j-2,k]
        f1 = DA[_by,i,j-1,k]
        f2 = DA[_by,i,j  ,k]  
        divBy =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    if k > 0 and k < nK-1: # in interior
        h1 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        h2 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        f0 = DA[_bz,i,j,k-1]
        f1 = DA[_bz,i,j,k  ]
        f2 = DA[_bz,i,j,k+1]
        divBz = - h2/h1/(h1+h2)*f0 - (h1-h2)/h1/h2*f1 + h1/h2/(h1+h2)*f2
    elif k == 0: # on face
        h1 = DA[_z ,i,j,k+1] - DA[_z,i,j,k  ]
        h2 = DA[_z ,i,j,k+2] - DA[_z,i,j,k+1]
        f0 = DA[_bz,i,j,k  ]
        f1 = DA[_bz,i,j,k+1]
        f2 = DA[_bz,i,j,k+2]
        divBz = - h1/h2/(h1+h2)*f2 + (h1+h2)/h1/h2*f1 - (2*h1+h2)/h1/(h1+h2)*f0        
    else: # k == nK-1: on face
        h1 = DA[_z ,i,j,k-1] - DA[_z,i,j,k-2]
        h2 = DA[_z ,i,j,k  ] - DA[_z,i,j,k-1]
        f0 = DA[_bz,i,j,k-2]
        f1 = DA[_bz,i,j,k-1]
        f2 = DA[_bz,i,j,k  ]
        divBz =   h2/h1/(h1+h2)*f0 - (h1+h2)/h1/h2*f1 + (2*h2+h1)/h2/(h1+h2)*f2   
                    
    divB = divBx + divBy + divBz
    
    return divB

@njit
def OpenGGCM_divBint_b(XGSM, timeISO, openggcm):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a OpenGGCM file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a volume integral over divergence of B.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in OpenGGCM file
              
        openggcm = OpenGGCM data
        
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
    """

    assert openggcm.DataArray.shape == (len(openggcm.varidx), openggcm.nI, openggcm.nJ, openggcm.nK)
    
    # Set up some variables used below
    B = np.zeros(3)
    r = np.zeros(3)
    
    # We use these values throughout the routine, so to avoid muliple lookups
    # we look for them once.
    _x = openggcm.varidx['x']
    _y = openggcm.varidx['y']
    _z = openggcm.varidx['z']
    
    # We calculate diVB in GSE to avoid numerical errors due to GSE->GSM transformation
    _xGSE = openggcm.varidx['xGSE']
    _yGSE = openggcm.varidx['yGSE']
    _zGSE = openggcm.varidx['zGSE']
    
    _bxGSE = openggcm.varidx['bxGSE']
    _byGSE = openggcm.varidx['byGSE']
    _bzGSE = openggcm.varidx['bzGSE']
    
    _measure = openggcm.varidx['measure']
    
    nVar, nI, nJ, nK = openggcm.DataArray.shape

    # Iterate thru points in simulation grid, calculating the divergence of B
    # at each point.  Use this in the divB integral from the Helmholtz
    # Decomposition Theorem to determine dB
    for i in range(nI):
        for j in range(nJ):
            for k in range(nK):
                
                # Distance from center of earth to point i,j,k,n
                r0 = np.sqrt(openggcm.DataArray[_x,i,j,k]**2 +
                             openggcm.DataArray[_y,i,j,k]**2 +
                             openggcm.DataArray[_z,i,j,k]**2)
                
                # Only include point if it is outside of rCurrents
                # Data are not valid inside rCurrents
                if r0 >= openggcm.rCurrents:
                    # Get divergence of B for integral
                    divB = calcDivB(openggcm.DataArray, i, j, k, nI, nJ, nK, 
                                    _xGSE, _yGSE, _zGSE, _bxGSE, _byGSE, _bzGSE)
                                        
                    # To calculate the integral, we need the distance from 
                    # point i,j,k to XGSM
                    r[0] = XGSM[0] - openggcm.DataArray[_x,i,j,k]
                    r[1] = XGSM[1] - openggcm.DataArray[_y,i,j,k]
                    r[2] = XGSM[2] - openggcm.DataArray[_z,i,j,k]
                    rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
                    
                    # dV for integral
                    measure = openggcm.DataArray[_measure,i,j,k]
                                
                    ##########################################################
                    # Below we calculate the delta B in each differential volume 
                    # element in the integral.  We want the final result to be 
                    # in nT.
                    # dB = 1/(4pi) divB x r/r^3 dV
                    #    = 1/(4pi) [nT/Re] [Re] / [Re^3] * [Re^3]
                    #    = 1/(4pi) with distances in Re, B in nT
                    ##########################################################
                    
                    B = B + divB * r * measure / rmag**3 / 4 / np.pi 
      
    return B

if __name__ == "__main__":

    import os.path

    ###############################################
    # Based on magnetopost info structure
    ###############################################

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
    
    nI        = oggcm.nI
    nJ        = oggcm.nJ
    nK        = oggcm.nK
    
    DataArray = oggcm.DataArray
    data_arr  = oggcm.data_arr
    
    _x = oggcm.varidx['x']
    _y = oggcm.varidx['y']
    _z = oggcm.varidx['z']

    _bx = oggcm.varidx['bxGSE']
    _by = oggcm.varidx['byGSE']
    _bz = oggcm.varidx['bzGSE']
    
    varidx = oggcm.varidx
    
    x = data_arr[:,_x]
    y = data_arr[:,_y]
    z = data_arr[:,_z]
    
    # # This should give us divB=0
    # data_arr[:,_bx] = 0
    # data_arr[:,_by] = 0
    # data_arr[:,_bz] = 0
    # value = 0.0
    
    # # This should give us divB=0
    # data_arr[:,_bx] = z
    # data_arr[:,_by] = x
    # data_arr[:,_bz] = y
    # value = 0.0

    # # This should give us divB=0
    # data_arr[:,_bx] = 10.*z**2
    # data_arr[:,_by] = 100.*x**2
    # data_arr[:,_bz] = 1000.*y**2
    # value = 0.0
    
    # # This should give us divB=0
    # data_arr[:,_bx] = 10.*y**2
    # data_arr[:,_by] = 100.*z**2
    # data_arr[:,_bz] = 1000.*x**2
    # value = 0.0
    
    # This should give us divB=3
    data_arr[:,_bx] = x
    data_arr[:,_by] = y
    data_arr[:,_bz] = z
    value = 3.0

    threshold = 0.0000001     

    # Calculate divB for each point on grid
    # Verify that we get the expected answer
    print( 'Check values')
    for i in range(nI):
        for j in range(nJ):
            for k in range(nK):
                divB = calcDivB(DataArray, i, j, k, nI, nJ, nK, _x, _y, _z, _bx, _by, _bz)
                if np.abs(divB - value) > threshold: print( i,j,k,divB )
    print('If no values shown, test passed')
    print('Done divB test')


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:38:31 2024

@author: Dean Thomas
"""

from numba import njit
import numpy as np

@njit
def calcDivB(batsrus, i, j, k, n, nI, nJ, nK, dX, dY, dZ, _bx, _by, _bz):
    """ Subroutine for calc_ms_divBint_b_sub that allows numba accelleration.  It  
    calculates divergence of B at point i,j,k in block n using data from 
    a BATSRUS file and the Helmholtz decompostion theorem to replace Biot-Savart 
    with a volume integral over the divergence of B.
    
    Inputs:
        batsrus = BATSRUS data
        
        i,j,k = grid coordinates of point inside block n
        
        nI,nJ,nK = number of x,y,z points, respectively, in block n. Provided to 
            avoid constantly looking them up
        
        dX,dY,dZ = distance between two consecutive points along x,y,z axis,
            respectively, for points inside block n.   Provided to avoid 
            constantly calculating them.
        
        _bx,_by,_bz = batsrus.varidx values for bx, by, bz.  Provided to avoid 
            constantly looking them up
                       
    Outputs:
        divB = divergence of B at point i,j,k in block n, which is sum of 
            divBx, divBy and divBz (in GSM coordinates)
    """
    
    assert( i>=0 and i<nI )
    assert( j>=0 and j<nJ )
    assert( k>=0 and k<nK )
    
    # Use 2nd order stencils to calculate derivatives, sum derivatives to 
    # determine divB

    if i > 0 and i < nI-1: # in interior of block n
        divBx = (batsrus.DataArray[_bx, i+1, j, k, n] - batsrus.DataArray[_bx, i-1, j, k, n])/(2*dX)
    elif i == 0: # on face
        divBx = (-3*batsrus.DataArray[_bx, 0, j, k, n] + 4*batsrus.DataArray[_bx, 1, j, k, n]
                - batsrus.DataArray[_bx, 2, j, k, n])/(2*dX)
    else: # i == nI-1: on face
        divBx = (3*batsrus.DataArray[_bx, nI-1, j, k, n] - 4*batsrus.DataArray[_bx, nI-2, j, k, n]
                + batsrus.DataArray[_bx, nI-3, j, k, n])/(2*dX)        
    
    if j > 0 and j < nJ-1: # in interior of block n
        divBy = (batsrus.DataArray[_by, i, j+1, k, n] - batsrus.DataArray[_by, i, j-1, k, n])/(2*dY)
    elif j == 0: # on face
        divBy = (-3*batsrus.DataArray[_by, i, 0, k, n] + 4*batsrus.DataArray[_by, i, 1, k, n]
                - batsrus.DataArray[_by, i, 2, k, n])/(2*dY)
    else: # j == nJ-1: on face
        divBy = (3*batsrus.DataArray[_by, i, nJ-1, k, n] - 4*batsrus.DataArray[_by, i, nJ-2, k, n]
                + batsrus.DataArray[_by, i, nJ-3, k, n])/(2*dY)        
    
    if k > 0 and k < nK-1: # in interior of block n
        divBz = (batsrus.DataArray[_bz, i, j, k+1, n] - batsrus.DataArray[_bz, i, j, k-1, n])/(2*dZ)
    elif k == 0: # on face
        divBz = (-3*batsrus.DataArray[_bz, i, j, 0, n] + 4*batsrus.DataArray[_bz, i, j, 1, n]
                - batsrus.DataArray[_bz, i, j, 2, n])/(2*dZ)
    else: # k == ni-1: on face
        divBz = (3*batsrus.DataArray[_bz, i, j, nK-1, n] - 4*batsrus.DataArray[_bz, i, j, nK-2, n]
                + batsrus.DataArray[_bz, i, j, nK-3, n])/(2*dZ)        
    
    divB = divBx + divBy + divBz
        
    return divB

@njit
def BATSRUS_divBint_b(XGSM, timeISO, batsrus):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a BATSRUS file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a volume integral over divergence of B.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data
        
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
    """

    # Set up some variables used below
    B = np.zeros(3)
    r = np.zeros(3)
    
    # We use these values throughout the routine, so to avoid muliple lookups
    # we look for them once.
    _x = batsrus.varidx['x']
    _y = batsrus.varidx['y']
    _z = batsrus.varidx['z']
    
    _bx = batsrus.varidx['bx']
    _by = batsrus.varidx['by']
    _bz = batsrus.varidx['bz']
    
    _measure = batsrus.varidx['measure']
    
    nVar, nI, nJ, nK, nBlock = batsrus.DataArray.shape

    # Loop through each block, then loop through each point in the block.
    for n in range(nBlock):
        
        # Determine dX, dY, and dZ for this block
        dX = batsrus.DataArray[_x,1,0,0,n] - batsrus.DataArray[_x,0,0,0,n]
        dY = batsrus.DataArray[_y,0,1,0,n] - batsrus.DataArray[_y,0,0,0,n]
        dZ = batsrus.DataArray[_z,0,0,1,n] - batsrus.DataArray[_z,0,0,0,n]
        
        # Iterate thru points in block, calculating the divergence of B at
        # each point.  Use this in the divB integral from the Helmholtz
        # Decomposition Theorem to determine dB
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):
                    # Distance from center of earth to point i,j,k,n
                    r0 = np.sqrt(batsrus.DataArray[_x,i,j,k,n]**2 +
                                 batsrus.DataArray[_y,i,j,k,n]**2 +
                                 batsrus.DataArray[_z,i,j,k,n]**2)
                    
                    # Only include point if it is outside of rCurrents
                    # Data are not valid inside rCurrents
                    if r0 >= batsrus.rCurrents:
                        # Get divergence of B for integral
                        divB = calcDivB(batsrus, i, j, k, n, nI, nJ, nK, 
                                        dX, dY, dZ, _bx, _by, _bz)
                        
                        # To calculate the integral, we need the distance from 
                        # point i,j,k,n to XGSM
                        r[0] = XGSM[0] - batsrus.DataArray[_x,i,j,k,n]
                        r[1] = XGSM[1] - batsrus.DataArray[_y,i,j,k,n]
                        r[2] = XGSM[2] - batsrus.DataArray[_z,i,j,k,n]
                        rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
                        
                        # dV for integral
                        measure = batsrus.DataArray[_measure,i,j,k,n]
                                    
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

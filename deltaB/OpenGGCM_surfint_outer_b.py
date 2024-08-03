#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:06:14 2024

@author: Dean Thomas
"""

import numba
import numpy as np

from deltaB.OpenGGCM_interpolator import OpenGGCM_interpolator

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
    
def OpenGGCM_surfint_outer_b(XGSM, timeISO, openggcm, nX=100, nY=100, nZ=100):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a OpenGGCM file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a surface integral on outer boundary of OpenGGCM grid.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in OpenGGCM file
              
        openggcm = OpenGGCM data
        
        nX, nY, nZ = number of steps in numerical integration over outer faces,
            e.g., nX*nY points on outer surfaces parallel to X-Y plane
                
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
        
        Birr, Bsol = irrotational and solenoidal components of B (GSM coordinates)
    """

    # Set up some variables used below
    B      = np.zeros(3)
    Bpt    = np.zeros(3)
    Birr   = np.zeros(3)
    Bsol   = np.zeros(3)
    
    # Create OpenGGCM interpolators, see openggcm_interpolator.py
    openggcm_interp = OpenGGCM_interpolator(openggcm)
    openggcm_interp.register_variable( 'bx' )
    openggcm_interp.register_variable( 'by' )
    openggcm_interp.register_variable( 'bz' )
    
    # We need the GSE to GSM transformation matrix below
    trans_mat = openggcm.GSE_to_GSM

    # Local routine that is used in loops below to calculate contribution
    # from each surface element.
    def calc( interpolator, xxGSE, xxhatGSE, dS ):
        """ xxGSE = point in space (GSE)
            xxhatGSE = unit vector for surface (GSE)
            dS = size of surface element
        """
        # Interpolator needs GSM coordinates and we get result in GSM
        xxGSM    = matmul( trans_mat, xxGSE )
        xxhatGSM = matmul( trans_mat, xxhatGSE )

        # Get B field at point xx (GSM) (result in GSM coordinates)
        Bpt[0] = interpolator.interpolator(xxGSM, 'bx')[0]
        Bpt[1] = interpolator.interpolator(xxGSM, 'by')[0]
        Bpt[2] = interpolator.interpolator(xxGSM, 'bz')[0]
            
        if( np.isnan(Bpt[0]) or np.isnan(Bpt[1]) or np.isnan(Bpt[2]) ):
            import sys
            sys.exit(f'B interpolation error: xx = {xxGSE}, xxhat = {xxhatGSE}, Bpt = {Bpt}')
                    
        # Distance to point XGSM where we want to know the magnetic field
        r = XGSM - xxGSM
        rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
        
        ##########################################################
        # Below we calculate the delta B in each differential surface 
        # element in the integral.  We want the final result to be in nT.
        # dB = 1/(4pi) B x r/r^3 dS
        #    = 1/(4pi) [nT] [Re] / [Re^3] * [Re^2]
        #    = 1/(4pi) with distances in Re, B in nT
        ##########################################################
    
        # Irrotational and solenodial contributions from Helmholtz decomposition
        # in GSM coordintates
        Birr[:] = Birr[:] - np.dot(Bpt,xxhatGSM) * r / rmag**3 * dS / 4 / np.pi
        Bsol[:] = Bsol[:] - np.cross( r, np.cross(Bpt,xxhatGSM) ) / rmag**3 * dS / 4 / np.pi
        return

    # Start the loops for surface numerical integration.  We will cover the 
    # six faces of the rectangular prism representing the outer boundary
    # of the OpenGGCM grid.  Note, the rectangular prism is in GSE, the original
    # coordinate system of the OpenGGCM data.   
    
    # Min/max in GSE coordinates
    minXGSE = openggcm.xGlobalMinGSE
    maxXGSE = openggcm.xGlobalMaxGSE
    minYGSE = openggcm.yGlobalMinGSE
    maxYGSE = openggcm.yGlobalMaxGSE
    minZGSE = openggcm.zGlobalMinGSE
    maxZGSE = openggcm.zGlobalMaxGSE
    
    # dX, dY, and dZ increments (GSE coordinates)
    dXGSE = (maxXGSE - minXGSE)/nX
    dYGSE = (maxYGSE - minYGSE)/nY
    dZGSE = (maxZGSE - minZGSE)/nZ

    # Differential surface area on each plane
    dSxy = dXGSE * dYGSE
    dSxz = dXGSE * dZGSE
    dSyz = dYGSE * dZGSE
    
    # loops for upper and lower faces (parallel to x-y plane)
    for i in range(nX):  
        # Find x at the middle of each differential surface element
        # from x - dX/2 to x + dX/2
        xloopGSE = minXGSE + (i + 0.5) * dXGSE
        
        for j in range(nY):
            # Find y at the middle of each differential surface element
            # from y - dY/2 to y + dY/2
            yloopGSE = minYGSE + (j + 0.5) * dYGSE
            
            # top face
            xGSE = np.array([xloopGSE, yloopGSE, maxZGSE])
            xhatGSE = np.array([0.,0.,1.])
            calc( openggcm_interp, xGSE, xhatGSE, dSxy )

            # bottom face
            xGSE = np.array([xloopGSE, yloopGSE, minZGSE])
            xhatGSE = np.array([0.,0.,-1.])
            calc( openggcm_interp, xGSE, xhatGSE, dSxy )
            
    # loops for left and right faces (parallel to x-z plane)
    for i in range(nX):  
        # Find x at the middle of each differential surface element
        # from x - dX/2 to x + dX/2
        xloopGSE = minXGSE + (i + 0.5) * dXGSE
        
        for j in range(nZ):
            # Find z at the middle of each differential surface element
            # from z - dZ/2 to z + dZ/2
            zloopGSE = minZGSE + (j + 0.5) * dZGSE
            
            # left face
            xGSE = np.array([xloopGSE, maxYGSE, zloopGSE])
            xhatGSE = np.array([0.,1.,0.])
            calc( openggcm_interp, xGSE, xhatGSE, dSxz )

            # right face
            xGSE = np.array([xloopGSE, minYGSE, zloopGSE])
            xhatGSE = np.array([0.,-1.,0.])
            calc( openggcm_interp, xGSE, xhatGSE, dSxz )

    # loops for front and back faces (parallel to y-z plane)
    for i in range(nY):  
        # Find y at the middle of each differential surface element
        # from y - dY/2 to y + dY/2
        yloopGSE = minYGSE + (i + 0.5) * dYGSE
        
        for j in range(nZ):
            # Find z at the middle of each differential surface element
            # from z - dZ/2 to z + dZ/2
            zloopGSE = minZGSE + (j + 0.5) * dZGSE
            
            # front face
            xGSE = np.array([maxXGSE, yloopGSE, zloopGSE])
            xhatGSE = np.array([1.,0.,0.])
            calc( openggcm_interp, xGSE, xhatGSE, dSyz )
            
            # back face
            xGSE = np.array([minXGSE, yloopGSE, zloopGSE])
            xhatGSE = np.array([-1.,0.,0.])
            calc( openggcm_interp, xGSE, xhatGSE, dSyz )
            
    # Add irrotational and solenoidal contributions to get total B contribution
    # in GSM coordinatees
    B[:] = Birr[:] + Bsol[:]
    
    return B, Birr, Bsol
  

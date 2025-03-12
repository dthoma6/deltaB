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
    BGSE      = np.zeros(3)
    BptGSE    = np.zeros(3)
    BirrGSE   = np.zeros(3)
    BsolGSE   = np.zeros(3)
    
    # Create OpenGGCM interpolators, see openggcm_interpolator.py
    # To avoid numerical errors due to GSE->GSM transformation, we use GSE coordinates
    openggcm_interp = OpenGGCM_interpolator(openggcm, GSMIN=False)
    openggcm_interp.register_variable( 'bxGSE' )
    openggcm_interp.register_variable( 'byGSE' )
    openggcm_interp.register_variable( 'bzGSE' )
    
    # We need the GSE to GSM transformation matrix below
    # trans_mat = openggcm.GSE_to_GSM
    trans_to_GSM = openggcm.GSE_to_GSM
    trans_to_GSE = openggcm.GSM_to_GSE
    
    # Need XGSM in GSE coordiantes below to calculate r
    XGSE = matmul( trans_to_GSE, XGSM )

    # Local routine that is used in loops below to calculate contribution
    # from each surface element.
    def calc( interpolator, xxGSE, xxhatGSE, dS ):
        """ xxGSE = point in space (GSE)
            xxhatGSE = unit vector for surface (GSE)
            dS = size of surface element
        """
        # Get B field at point xx (GSE) (result in GSE coordinates)
        BptGSE[0] = interpolator.interpolator(xxGSE, 'bxGSE')[0]
        BptGSE[1] = interpolator.interpolator(xxGSE, 'byGSE')[0]
        BptGSE[2] = interpolator.interpolator(xxGSE, 'bzGSE')[0]

        if( np.isnan(BptGSE[0]) or np.isnan(BptGSE[1]) or np.isnan(BptGSE[2]) ):
            import sys
            sys.exit(f'B interpolation error: xx = {xxGSE}, xxhat = {xxhatGSE}, Bpt = {BptGSE}')
                    
        # Distance to point XGSE where we want to know the magnetic field
        r = XGSE - xxGSE
        rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
        
        ##########################################################
        # Below we calculate the delta B in each differential surface 
        # element in the integral.  We want the final result to be in nT.
        # dB = 1/(4pi) B x r/r^3 dS
        #    = 1/(4pi) [nT] [Re] / [Re^3] * [Re^2]
        #    = 1/(4pi) with distances in Re, B in nT
        ##########################################################
    
        # Irrotational and solenodial contributions from Helmholtz decomposition
        # in GSE coordintates
        BirrGSE[:] = BirrGSE[:] - np.dot(BptGSE,xxhatGSE) * r / rmag**3 * dS / 4 / np.pi
        BsolGSE[:] = BsolGSE[:] - np.cross( r, np.cross(BptGSE,xxhatGSE) ) / rmag**3 * dS / 4 / np.pi

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
    # in GSE coordinatees
    BGSE[:] = BirrGSE[:] + BsolGSE[:]
    
    # Transform to GSM coordinates
    B    = matmul( trans_to_GSM, BGSE )
    Birr = matmul( trans_to_GSM, BirrGSE )
    Bsol = matmul( trans_to_GSM, BsolGSE )

    return B, Birr, Bsol
  

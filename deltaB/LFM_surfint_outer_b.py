#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:17:12 2024

@author: Dean Thomas
"""

import numba
import numpy as np

# Boolean to determine which interpolator is used
if True:
    from deltaB.LFM_interpolator import LFM_interpolator
else:
    from deltaB.LFM_interpolator2 import LFM_interpolator2 as LFM_interpolator
    
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
    
def LFM_surfint_outer_b(XGSM, timeISO, lfm, nX=100, nR=100, nTheta=100):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a LFM file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a surface integral on outer boundary of LFM grid.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in LFM file
              
        LFM = LFM data
        
        nX, nR, nTheta = number of steps in numerical integration over outer faces,
            e.g., nX*nTheta points on outer cylindrical wall
        
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
        
        Birr, Bsol = irrotational and solenoidal components of B (GSM coordinates)
    """

    # Set up some variables used below
    B      = np.zeros(3)
    Bpt    = np.zeros(3)
    Birr   = np.zeros(3)
    Bsol   = np.zeros(3)
    
    # Create LFM interpolators, see LFM_interpolator.py
    LFM_interp = LFM_interpolator(lfm)
    LFM_interp.register_variable( 'bx' )
    LFM_interp.register_variable( 'by' )
    LFM_interp.register_variable( 'bz' )
    
    # We need the SM to GSM transformation matrix below
    trans_mat = lfm.SM_to_GSM

    # Local routine that is used in loops below to calculate contribution
    # from each surface element.
    def calc( interpolator, xxSM, xxhatSM, dS ):
        """ xxSM = point in space (SM)
            xxhatSM = unit vector for surface (SM)
            dS = size of surface element
        """
        # Interpolator needs GSM coordinates and we get result in GSM
        xxGSM    = matmul( trans_mat, xxSM )
        xxhatGSM = matmul( trans_mat, xxhatSM )

        # Get B field at point xx (GSM) (result in GSM coordinates)
        Bpt[0] = interpolator.interpolator(xxGSM, 'bx')[0]
        Bpt[1] = interpolator.interpolator(xxGSM, 'by')[0]
        Bpt[2] = interpolator.interpolator(xxGSM, 'bz')[0]
            
        if( np.isnan(Bpt[0]) or np.isnan(Bpt[1]) or np.isnan(Bpt[2]) ):
            import sys
            sys.exit(f'B interpolation error: xx = {xxSM}, xxhat = {xxhatSM}, Bpt = {Bpt}')
                    
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

    # Start the loops for surface numerical integration.  The outer boundary is
    # a cylinder.  We will cover the cylindrical wall and the two end caps
    # Note, the cylinder is in SM, the original coordinate system of the LFM data.   
    
    # Min/max in SM coordinates
    minXSM = np.min(lfm.cellverticesSM[:,0]) # end caps at max/min XSM
    maxXSM = np.max(lfm.cellverticesSM[:,0])
    
    ySM = lfm.cellverticesSM[:,1]
    zSM = lfm.cellverticesSM[:,2]
    rSM = np.sqrt( ySM**2 + zSM**2 )
    maxRSM = np.max( rSM )
    
    # dX, dY, and dZ increments (SM coordinates)
    dXSM = (maxXSM - minXSM)/nX
    dTSM = (2.*np.pi)/nTheta
    dRSM = (maxRSM)/nR

    # loops for outer cylindrical wall
    for i in range(nX):  
        # Find x at the middle of each differential surface element
        # from x - dX/2 to x + dX/2
        xloopSM = minXSM + (i + 0.5) * dXSM
        
        for j in range(nTheta):
            # Find theta at the middle of each differential surface element
            # from theta - dtheta/2 to theta + dtheta/2
            tloopSM = (j + 0.5) * dTSM
            
            yloopSM = maxRSM * np.cos(tloopSM)
            zloopSM = maxRSM * np.sin(tloopSM)
            
            dS = dXSM * maxRSM * dTSM
            
            xSM = np.array([xloopSM, yloopSM, zloopSM])
            xhatSM = np.array([0.,np.cos(tloopSM),np.sin(tloopSM)])
            calc( LFM_interp, xSM, xhatSM, dS )

    # loops for end caps
    for i in range(nR):  
        # Find r at the middle of each differential surface element
        # from r - dr/2 to r + dR/2
        rloopSM = (i + 0.5) * dRSM
        
        for j in range(nTheta):
            # Find theta at the middle of each differential surface element
            # from theta - dtheta/2 to theta + dtheta/2
            tloopSM = (j + 0.5) * dTSM
            
            yloopSM = rloopSM * np.cos(tloopSM)
            zloopSM = rloopSM * np.sin(tloopSM)
            
            dS = dRSM * rloopSM * dTSM
            
            # left end cap
            xSM = np.array([minXSM, yloopSM, zloopSM])
            xhatSM = np.array([-1.,0.,0.])
            calc( LFM_interp, xSM, xhatSM, dS )

            # right end cap
            xSM = np.array([maxXSM, yloopSM, zloopSM])
            xhatSM = np.array([1.,0.,0.])
            calc( LFM_interp, xSM, xhatSM, dS )
            
    # Add irrotational and solenoidal contributions to get total B contribution
    # in GSM coordinatees
    B[:] = Birr[:] + Bsol[:]
    
    return B, Birr, Bsol
  

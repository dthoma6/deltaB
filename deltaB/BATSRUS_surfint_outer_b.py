#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:06:14 2024

@author: Dean Thomas
"""

import numpy as np

KAMODO=False # Use Kamodo interpolator or swmfio interpolator
if KAMODO:
    from deltaB.BATSRUS_interpolator import BATSRUS_interpolator
else:
    from deltaB.BATSRUS_interpolator2 import BATSRUS_interpolator2

def BATSRUS_surfint_outer_b(XGSM, timeISO, batsrus, nX=100, nY=100, nZ=100):
    """ Subroutine for calc_ms_surfint_b.  It calculates total B field at point 
    XGSM using data from a BATSRUS file and the Helmholtz decompostion theorem 
    to replace Biot-Savart volume integral with a surface integral on outer 
    boundary of BATSRUS grid.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data
        
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
    
    if KAMODO:
        # Create Kamodo BATSRUS interpolators, see BATSRUS_interpolator.py
        batsrus_interp = BATSRUS_interpolator(batsrus)
        batsrus_interp.register_variable( 'bx' )
        batsrus_interp.register_variable( 'by' )
        batsrus_interp.register_variable( 'bz' )
    else:
        # Create swmfio-based BATSRUS interpolators, see BATSRUS_interpolator2.py
        batsrus_interp = BATSRUS_interpolator2(batsrus)
        batsrus_interp.register_variable( 'bx' )
        batsrus_interp.register_variable( 'by' )
        batsrus_interp.register_variable( 'bz' )
    
    # Local routine that is used in loops below to calculate contribution
    # from each surface element.
    def calc( interpolator, xx, xxhat, dS ):
        """ xx = point in space (GSM)
            xxhat = unit vector for surface
            dS = size of surface element
        """
        # Get B field at point xx (in GSM coordinates)
        if KAMODO:
            FACTOR = 0.9 # To avoid interpolator errors at boundary
            Bpt[0] = interpolator.interpolator(FACTOR*xx, 'bx')[0]
            Bpt[1] = interpolator.interpolator(FACTOR*xx, 'by')[0]
            Bpt[2] = interpolator.interpolator(FACTOR*xx, 'bz')[0]
        else:
            Bpt[0] = interpolator.interpolator(xx, 'bx')
            Bpt[1] = interpolator.interpolator(xx, 'by')
            Bpt[2] = interpolator.interpolator(xx, 'bz')
            
        if( np.isnan(Bpt[0]) or np.isnan(Bpt[1]) or np.isnan(Bpt[2]) ):
            import sys
            sys.exit(f'B interpolation error: xx = {xx}, xxhat = {xxhat}, Bpt = {Bpt}')
                    
        # Distance to point XGSM where we want to know the magnetic field
        r = XGSM - xx
        rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
        
        ##########################################################
        # Below we calculate the delta B in each differential surface 
        # element in the integral.  We want the final result to be in nT.
        # dB = 1/(4pi) B x r/r^3 dS
        #    = 1/(4pi) [nT] [Re] / [Re^3] * [Re^2]
        #    = 1/(4pi) with distances in Re, B in nT
        ##########################################################
    
        # Irrotational and solenodial contributions from Helmholtz decomposition
        Birr[:] = Birr[:] - np.dot(Bpt,xxhat) * r / rmag**3 * dS / 4 / np.pi
        Bsol[:] = Bsol[:] - np.cross( r, np.cross(Bpt,xxhat) ) / rmag**3 * dS / 4 / np.pi
        return

    # Start the loops for surface numerical integration.  We will cover the 
    # six faces of the rectangular prism representing the outer boundary
    # of the BATSRUS grid
    
    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)
    
    minX = np.min(batsrus.data_arr[:, var_dict['x']][:])
    maxX = np.max(batsrus.data_arr[:, var_dict['x']][:])
    minY = np.min(batsrus.data_arr[:, var_dict['y']][:])
    maxY = np.max(batsrus.data_arr[:, var_dict['y']][:])
    minZ = np.min(batsrus.data_arr[:, var_dict['z']][:])
    maxZ = np.max(batsrus.data_arr[:, var_dict['z']][:])
    
    # dX, dY, and dZ increments (GSM coordinates)
    dX = (maxX - minX)/nX
    dY = (maxY - minY)/nY
    dZ = (maxZ - minZ)/nZ

    # Differential surface area on each plane
    dSxy = dX*dY
    dSxz = dX*dZ
    dSyz = dY*dZ

    # loops for upper and lower faces (parallel to x-y plane)
    for i in range(nX):  
        # Find x at the middle of each differential surface element
        # from x - dX/2 to x + dX/2
        xloop = minX + (i + 0.5) * dX
        
        for j in range(nY):
            # Find y at the middle of each differential surface element
            # from y - dY/2 to y + dY/2
            yloop = minY + (j + 0.5) * dY
            
            # top face
            x = np.array([xloop, yloop, maxZ])
            xhat = np.array([0.,0.,1.])
            calc( batsrus_interp, x, xhat, dSxy )

            # bottom face
            x = np.array([xloop, yloop, minZ])
            xhat = np.array([0.,0.,-1.])
            calc( batsrus_interp, x, xhat, dSxy )
            
    # loops for left and right faces (parallel to x-z plane)
    for i in range(nX):  
        # Find x at the middle of each differential surface element
        # from x - dX/2 to x + dX/2
        xloop = minX + (i + 0.5) * dX
        
        for j in range(nZ):
            # Find z at the middle of each differential surface element
            # from z - dZ/2 to z + dZ/2
            zloop = minZ + (j + 0.5) * dZ
            
            # left face
            x = np.array([xloop, maxY, zloop])
            xhat = np.array([0.,1.,0.])
            calc( batsrus_interp, x, xhat, dSxz )

            # right face
            x = np.array([xloop, minY, zloop])
            xhat = np.array([0.,-1.,0.])
            calc( batsrus_interp, x, xhat, dSxz )

    # loops for front and back faces (parallel to y-z plane)
    for i in range(nY):  
        # Find y at the middle of each differential surface element
        # from y - dY/2 to y + dY/2
        yloop = minY + (i + 0.5) * dY
        
        for j in range(nZ):
            # Find z at the middle of each differential surface element
            # from z - dZ/2 to z + dZ/2
            zloop = minZ + (j + 0.5) * dZ
            
            # front face
            x = np.array([maxX, yloop, zloop])
            xhat = np.array([1.,0.,0.])
            calc( batsrus_interp, x, xhat, dSyz )
            
            # back face
            x = np.array([minX, yloop, zloop])
            xhat = np.array([-1.,0.,0.])
            calc( batsrus_interp, x, xhat, dSyz )
            
    # Add irrotational and solenoidal contributions to get total B contribution
    B[:] = Birr[:] + Bsol[:]
    
    # We no longer need the interpolator, and deleting it avoids memory error
    del batsrus_interp
    
    return B, Birr, Bsol
  

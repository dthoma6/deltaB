#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:14:30 2024

@author: Dean Thomas
"""

import numpy as np 

KAMODO=False  # Use Kamodo interpolator or swmfio interpolator
if KAMODO:
    from deltaB.BATSRUS_interpolator import BATSRUS_interpolator
else:
    from deltaB.BATSRUS_interpolator2 import BATSRUS_interpolator2

def BATSRUS_surfint_rCurrents_b(XGSM, timeISO, batsrus, nTheta=180, nPhi=180):
    """ Subroutine for calc_ms_surfint_rCurrents_b.
    It calculates total B field at point XGSM using data from a BATSRUS file 
    and the Helmholtz decompostion theorem to replace Biot-Savart volume integral 
    with a surface integral at rCurrents.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data from swmfio
        
        nTheta, nPhi = number of steps in numerical integration over theta
            and phi in the surface integral over a sphere at rCurrents
                
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
        
        Birr, Bsol = irrotational and solenoidal components of B (GSM coordinates)
    """

    # Set up some variables used below
    B      = np.zeros(3)
    r      = np.zeros(3)
    Bpt    = np.zeros(3)
    x      = np.zeros(3)
    xhat   = np.zeros(3)
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
    
    # Start the loops for surface numerical integration. We use two 
    # loops, theta and phi, which cover the inner boundary of the
    # magnetosphere (a sphere at rCurrents).
    
    # theta increments and phi increments (GSM coordinates)
    dTheta = np.pi/nTheta
    dPhi = 2. * np.pi/nPhi

    # theta loop, theta pi/2 -> -pi/2
    for i in range(nTheta):  
        # Find theta at the middle of each differential surface element
        # from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - (i + 0.5) * dTheta

        # Differential surface area on sphere at rCurrents
        dS = batsrus.rCurrents**2 * np.cos( theta ) * dTheta * dPhi
        
        # phi loop, phi 0 -> 2pi 
        for j in range(nPhi): 
            # Find phi at the middle of each differential surface element
            # from phi - dPhi/2 to phi + dPhi/2
            phi = (j + 0.5) * dPhi
        
            # Normal unit vector on sphere at rCurrents (GSM coordinates)
            # Unit vector points radially for gap region
            xhat[0] = np.cos( theta ) * np.cos( phi )
            xhat[1] = np.cos( theta ) * np.sin( phi )
            xhat[2] = np.sin( theta )
          
            # Point on sphere at rCurrents (GSM coordinates)
            x = xhat * batsrus.rCurrents
            
            # Get B field at point x (in GSM coordinates)
            if KAMODO:
                Bpt[0] = batsrus_interp.interpolator(x, 'bx')[0]
                Bpt[1] = batsrus_interp.interpolator(x, 'by')[0]
                Bpt[2] = batsrus_interp.interpolator(x, 'bz')[0]
            else:
                Bpt[0] = batsrus_interp.interpolator(x, 'bx')
                Bpt[1] = batsrus_interp.interpolator(x, 'by')
                Bpt[2] = batsrus_interp.interpolator(x, 'bz')

            # Distance to point XGSM where we want to know the magnetic field
            r = XGSM - x
            rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
            
            ##########################################################
            # Below we calculate the delta B in each differential surface 
            # element in the integral.  We want the final result to be in nT.
            # dB = 1/(4pi) B x r/r^3 dS
            #    = 1/(4pi) [nT] [Re] / [Re^3] * [Re^2]
            #    = 1/(4pi) with distances in Re, B in nT
            ##########################################################
    
            # Irrotational and solenodial contributions from Helmholtz decomposition
            Birr[:] = Birr[:] - np.dot(Bpt,xhat) * r / rmag**3 * dS / 4 / np.pi
            Bsol[:] = Bsol[:] - np.cross( r, np.cross(Bpt,xhat) ) / rmag**3 * dS / 4 / np.pi
                             
    # Add irrotational and solenoidal contributions to get total B contribution
    B[:] = Birr[:] + Bsol[:]
    
    # We no longer need the interpolator, and deleting it avoids memory error
    del batsrus_interp

    return B, Birr, Bsol

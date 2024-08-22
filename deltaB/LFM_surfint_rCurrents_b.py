#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:07:58 2024

@author: Dean Thomas
"""

import numpy as np 

# Boolean to determine which interpolator is used
if True:
    from deltaB.LFM_interpolator import LFM_interpolator
else:
    from deltaB.LFM_interpolator2 import LFM_interpolator2 as LFM_interpolator

def LFM_surfint_rCurrents_b(XGSM, timeISO, lfm, nTheta=180, nPhi=180):
    """ Subroutine for calc_ms_surfint_rCurrents_b.
    It calculates total B field at point XGSM using data from a LFM file 
    and the Helmholtz decompostion theorem to replace Biot-Savart volume integral 
    with a surface integral at rCurrents.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        lfm = LFM data
        
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
    
    # Create LFM interpolators, see LFM_interpolator.py
    LFM_interp = LFM_interpolator(lfm)
    LFM_interp.register_variable( 'bx' )
    LFM_interp.register_variable( 'by' )
    LFM_interp.register_variable( 'bz' )
    
    # Start the loops for surface numerical integration. We use two 
    # loops, theta and phi, which cover the inner boundary of the
    # magnetosphere (a sphere at rCurrents).
    
    # theta increments and phi increments (GSM coordinates)
    dTheta = np.pi/nTheta
    dPhi = 2. * np.pi/nPhi
    
    rCurrents = lfm.rCurrents

    # theta loop, theta pi/2 -> -pi/2
    for i in range(nTheta):  
        # Find theta at the middle of each differential surface element
        # from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - (i + 0.5) * dTheta

        # Differential surface area on sphere at rCurrents
        dS = rCurrents**2 * np.cos( theta ) * dTheta * dPhi
        
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
            x = xhat * rCurrents 
            
            # Get B field at point x (in GSM coordinates)
            Bpt[0] = LFM_interp.interpolator(x, 'bx')[0]
            Bpt[1] = LFM_interp.interpolator(x, 'by')[0]
            Bpt[2] = LFM_interp.interpolator(x, 'bz')[0]

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
    
    return B, Birr, Bsol

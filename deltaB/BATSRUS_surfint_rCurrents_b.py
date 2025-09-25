#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:14:30 2024

@author: Dean Thomas
"""

import numpy as np 

KAMODO=False  # Use Kamodo interpolator or swmfio interpolator, swmfio preferred
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
                Bpt[0] = batsrus_interp.interpolator(x, 'bx')[0]
                Bpt[1] = batsrus_interp.interpolator(x, 'by')[0]
                Bpt[2] = batsrus_interp.interpolator(x, 'bz')[0]

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

if __name__ == "__main__":

    # Code below tests BATSRUS_surfint_rCurrents_b against a known B field

    import os.path

    # Read in a BATSRUS file
   
    data_dir = r'/Volumes/PhysicsHD'
    # data_dir = r'/Volumes/Data1'
    
    info = {
            "model": "SWMF",
            "run_name": "Bob_Weigel_070323_3",
            # "rCurrents": 3.0,
            "rIonosphere": 1.01725,
            "file_type": "cdf",
            "method": "method1",
            "dir_run": os.path.join(data_dir, "Bob_Weigel_070323_3"),
            "dir_plots": os.path.join(data_dir, "Bob_Weigel_070323_3.plots"),
            "dir_derived": os.path.join(data_dir, "Bob_Weigel_070323_3.derived"),
            "dir_magnetosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "GM_CDF"),
            "dir_ionosphere": os.path.join(data_dir, "Bob_Weigel_070323_3", "IONO-2D_CDF")
    }
    
    file = '/Volumes/PhysicsHD/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-001800-000.out.cdf'
              
    from deltaB import get_batsrus_data_from_cdf

    bats = get_batsrus_data_from_cdf(file, info)
    
    DataArray = bats.DataArray
    data_arr  = bats.data_arr
    
    _x = bats.varidx['x']
    _y = bats.varidx['y']
    _z = bats.varidx['z']

    _bx = bats.varidx['bx']
    _by = bats.varidx['by']
    _bz = bats.varidx['bz']
    
    varidx = bats.varidx
    
    x = data_arr[:,_x]
    y = data_arr[:,_y]
    z = data_arr[:,_z]
    
    XGSM = np.array([0.,0.,0.])
    timeISO = '2000-01-01T00:18:00'  # ISO rime for file above
    
    # Replace B field from file with known B field
    # This should give us outer integral of the same values for B
    valuex = 10.0
    valuey = 100.0
    valuez = 1000.0
    data_arr[:,_bx] = valuex
    data_arr[:,_by] = valuey
    data_arr[:,_bz] = valuez

    # threshold = 0.0000001     

    # Calculate outer integral
    # Verify that we get the expected answer
    print('Check values')
    B, Birr, Bsol = BATSRUS_surfint_rCurrents_b(XGSM, timeISO, bats)
    print( 'Expect: ', np.array([valuex, valuey, valuez]) )
    print( 'Found: ', B )
    print('Done inner integral test')


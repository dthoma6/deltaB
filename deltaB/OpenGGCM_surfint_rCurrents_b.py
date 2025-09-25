#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:14:30 2024

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

def OpenGGCM_surfint_rCurrents_b(XGSM, timeISO, openggcm, nTheta=180, nPhi=180):
    """ Subroutine for calc_ms_surfint_rCurrents_b.
    It calculates total B field at point XGSM using data from a OpenGGCM file 
    and the Helmholtz decompostion theorem to replace Biot-Savart volume integral 
    with a surface integral at rCurrents.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in OpenGGCM file
              
        openggcm = OpenGGCM data
        
        nTheta, nPhi = number of steps in numerical integration over theta
            and phi in the surface integral over a sphere at rCurrents
                
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
        
        Birr, Bsol = irrotational and solenoidal components of B (GSM coordinates)
    """

    # Set up some variables used below
    BGSE      = np.zeros(3)
    BptGSE    = np.zeros(3)
    BirrGSE   = np.zeros(3)
    BsolGSE   = np.zeros(3)
    r         = np.zeros(3)
    xxGSE     = np.zeros(3)
    xxhatGSE  = np.zeros(3)

    # Create OpenGGCM interpolators, see openggcm_interpolator.py
    # To avoid GSE->GSM numerical errors, we'll do everything in GSE
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

    # Start the loops for surface numerical integration. We use two 
    # loops, theta and phi, which cover the inner boundary of the
    # magnetosphere (a sphere at rCurrents).
    
    # theta increments and phi increments (GSE coordinates)
    dTheta = np.pi/nTheta
    dPhi = 2. * np.pi/nPhi

    # theta loop, theta pi/2 -> -pi/2
    for i in range(nTheta):  
        # Find theta at the middle of each differential surface element
        # from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - (i + 0.5) * dTheta

        # Differential surface area on sphere at rCurrents
        dS = openggcm.rCurrents**2 * np.cos( theta ) * dTheta * dPhi
        
        # phi loop, phi 0 -> 2pi 
        for j in range(nPhi): 
            # Find phi at the middle of each differential surface element
            # from phi - dPhi/2 to phi + dPhi/2
            phi = (j + 0.5) * dPhi
        
            # Normal unit vector on sphere at rCurrents (GSE coordinates)
            # Unit vector points radially for gap region
            xxhatGSE[0] = np.cos( theta ) * np.cos( phi )
            xxhatGSE[1] = np.cos( theta ) * np.sin( phi )
            xxhatGSE[2] = np.sin( theta )

            # Point on sphere at rCurrents (GSE coordinates)
            xxGSE = xxhatGSE * openggcm.rCurrents
            
            # Get B field at point x (in GSM coordinates)
            BptGSE[0] = openggcm_interp.interpolator(xxGSE, 'bxGSE')[0]
            BptGSE[1] = openggcm_interp.interpolator(xxGSE, 'byGSE')[0]
            BptGSE[2] = openggcm_interp.interpolator(xxGSE, 'bzGSE')[0]
            
            # Distance to point XGSM where we want to know the magnetic field
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
            BirrGSE[:] = BirrGSE[:] - np.dot(BptGSE,xxhatGSE) * r / rmag**3 * dS / 4 / np.pi
            BsolGSE[:] = BsolGSE[:] - np.cross( r, np.cross(BptGSE,xxhatGSE) ) / rmag**3 * dS / 4 / np.pi
                             
    # Add irrotational and solenoidal contributions to get total B contribution
    BGSE[:] = BirrGSE[:] + BsolGSE[:]
    
    # Transform to GSM coordinates
    B    = matmul( trans_to_GSM, BGSE )
    Birr = matmul( trans_to_GSM, BirrGSE )
    Bsol = matmul( trans_to_GSM, BsolGSE )

    return B, Birr, Bsol

if __name__ == "__main__":

    # Code below tests OpenGGCM_surfint_rCurrents_b against a known B field

    import os.path

    # Read in a OpenGGCM file
    
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
    
    XGSM = np.array([0.,0.,0.])
    timeISO = '2000-01-01T04:30:00'  # ISO time for file 
    
    # Replace B field from file with known B field
    # This should give us outer integral of the same values for B
    valuex = 10.0
    valuey = 100.0
    valuez = 1000.0
    data_arr[:,_bx] = valuex
    data_arr[:,_by] = valuey
    data_arr[:,_bz] = valuez

    # # Replace B field from file with known B field
    # # This should give us outer integral of 0,0,0 at the origin
    # valuex = 0.
    # valuey = 0.
    # valuez = 0.
    # data_arr[:,_bx] = 0.
    # data_arr[:,_by] = 10.*z
    # data_arr[:,_bz] = 10.*y
    
    # # Replace B field from file with known B field
    # # This should give us outer integral of 0,0,0 at the origin
    # valuex = 0.
    # valuey = 0.
    # valuez = 0.
    # data_arr[:,_bx] = 10.*z
    # data_arr[:,_by] = 0
    # data_arr[:,_bz] = 10.*x

    threshold = 0.0000001     

    # Calculate outer integral
    # Verify that we get the expected answer
    print('Check values')
    B, Birr, Bsol = OpenGGCM_surfint_rCurrents_b(XGSM, timeISO, oggcm)
    print( 'Expect: ', np.array([valuex, valuey, valuez]) )
    print( 'Found: ', B )
    print('Done inner integral test')


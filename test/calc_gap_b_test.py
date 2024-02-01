#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 13:47:11 2023

@author: Dean Thomas
"""
import logging
import os.path
import numpy as np
from copy import deepcopy

from deltaB import calc_gap_b_sub, calc_gap_b_rim_sub, date_timeISO, get_NED_components

def test_calc_gap_b( rCurrents=3., rIonosphere=1.01727, nR=800):
    """Process data in RIM file to calculate the delta B using the test defined
    in the paper - CalcDeltaB: An efficient postprocessing tool to calculate 
    ground-level magnetic perturbations from global magnetosphere simulations 
    by Lutz Rastätter, Gábor Tóth, Maria M. Kuznetso.  
    
    Section 6.3 defines a test with an analytic solution.  Be at YKC is 
    0.3323795 based on a 10 kA current through the north pole.  We simulate 
    that scenario.  
    
    Inputs:
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nR = number of points to be examined in the numerical integration. nR 
            points in spherical grid between rIonosphere and rCurrents
                        
    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    ############################################################
    # Earth dipole field is in SM coordinates, and RIM is in SM
    # cordinates, so we do everything in SM coordinates
    ############################################################

    # Determine the size of the steps in r for numerical integration
    # over spherical shell between rIonosphere and rCurrents
    dR = (rCurrents - rIonosphere) / nR
    
    ##########################################################################
    # Per the CalcDeltaB paper, create jr grid, the radial component of the 
    # current density.  Grid spacing defined by 1 deg in latitude and 2 degrees 
    # in longitude
    ##########################################################################
    
    # theta array is a repeating array going from 0 to 89 degrees in northern
    # hemisphere.
    theta_tmpN = np.linspace(1,89,89) * np.pi/180

    # theta array repeats from 90 to 179 degrees for equator and southern
    # hemisphere
    theta_tmpS = np.linspace(90,179,90) * np.pi/180
      
    # phi array is a repeating set of repeating values. e.g., 180 0's followed
    # by 180 1's, etc. We will create this using a ones array
    phi_tmpN = np.ones(89)
    phi_tmpS = np.ones(90)
    
    # First two entries are the poles at 0 and pi
    theta_array = np.zeros(2)
    theta_array[1] = np.pi
    phi_array = np.zeros(2)
    
    # Concatenate to generate remaining parts of theta and phi arrays.
    for i in range(180):
        theta_array = np.concatenate((theta_array, theta_tmpN))
        phi_array = np.concatenate((phi_array, 2*i*phi_tmpN*np.pi/180 ))
    
    for i in range(180):
        theta_array = np.concatenate((theta_array, theta_tmpS))
        phi_array = np.concatenate((phi_array, 2*i*phi_tmpS*np.pi/180 ))
   
    # To populate jr_array, we want all of the cells at 0 to 1 degrees theta to have a
    # current density of 1 microamp/m^2 to match the test in CalcDeltaB paper
    # The paper wants a total current of 10000 amps along the north pole axis
    # so we're putting 10000 amps over an areas of 2 pi r^2 sin(dTheta) dTheta
    #
    # Note: use sin(dTheta/2) to get radius to the middle of the integration region 
    # Note: 10^6 to get microamps/m^2
    
    # Current through the north pole only, per scenario in paper
    jr = 1. # As specified in paper, yields 10kA current
    jr_array = np.zeros(32222) # 2 poles + 89*180 northern hemi + 90*180 equator and southern hemi
    jr_array[0] = jr

    # Two optional scenarios based on the scenario in the paper.  Answer should
    # be close to the same answer, but not quite.  They're approximations to 
    # scenaro in paper.
    
    # # Optional scenario 1:
    # # Ring around pole at 1 deg from north pole, additional scenario for testing
    # jr = 10000. / rIonosphere**2 / 6371000.**2 / dTheta / (2*np.pi) / np.sin(dTheta) * 10**6 
    # jr_array = np.zeros(32222) # 2 poles + 89*180 northern + 90*180 equator and southern
    # for i in range(180):
    #     jr_array[2+i*89] = jr
    
    # # Optional scenario 2:
    # # Pole and ring around pole at 1 deg from pole, additional scenario for testing
    # jrP = 5000. / rIonosphere**2 / 6371000.**2 / (dTheta/2)**2 / (np.pi) * 10**6 
    # jrR = 5000. / rIonosphere**2 / 6371000.**2 / dTheta / (2*np.pi) / np.sin(dTheta) * 10**6 
    # jr_array = np.zeros(32222) # 2 poles + 89*180 northern + 90*180 equator and southern
    # jr_array[0] = jrP
    # for i in range(180):
    #     jr_array[2+i*89] = jrR

    # Get the magnetometer location using list in magnetopost
    # from magnetopost.config import defined_magnetometers
    import magnetopost.config as mpc
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    # Change to YKC in CalcDeltaB paper, which is slightly different from
    # YKC in magnetopost
    pointX = mpc.Magnetometer(name='YKC-CalcDeltaB',csys='MAG',ctype='sph',
                              coords=(1.,  68.93,  302.49 ) )
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # We need the ISO time to update the magnetometer position
    timeISO = date_timeISO( (2011,1,1,12,0,0) )
    
    # Get the magnetometer position, X, in SM coordinates for compatibility with
    # RIM data
    XGEO.ticks = Ticktock([timeISO], 'ISO')
    XSM = XGEO.convert( 'SM', 'car' )
    X = XSM.data[0]

    # Old method of calculating delta B
    # B = calc_gap_b_sub(X, timeISO, rCurrents, rIonosphere, nTheta, nPhi, nR,
    #                     dTheta, dPhi, dR, theta_array, phi_array, jr_array)

    # New method of calculating delta B based on reverse engineered format
    # of RIM files
    B = calc_gap_b_rim_sub(X, timeISO, rCurrents, rIonosphere, nR,
                        dR, theta_array, phi_array, jr_array)
    
    Bn, Be, Bd = get_NED_components( B, X )
    
    print('nR: ', nR)
    print('Bxyz: ', B)    
    print('Bned: ', Bn, Be, Bd)
    
    return Bn, Be, Bd, B[0], B[1], B[2]      

if __name__ == "__main__":
    Bn, Be, Bd, Bx, By, Bz = test_calc_gap_b(nR=800)
    print('Analytic value for Be: 0.3323795 nT')
    print ('Our Be value is: ', Be, ' nT')
    
    
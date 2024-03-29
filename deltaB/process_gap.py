#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:10:35 2023

@author: Dean Thomas
"""

from numba import jit
import logging
import numpy as np
import swmfio
import pandas as pd
from datetime import datetime, timedelta
from spacepy.pybats.rim import Iono
from spacepy.time import Ticktock
import os.path

# from deltaB.coordinates import get_NED_vector_components
from deltaB.util import create_directory, get_NED_components, date_timeISO

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

@jit(nopython=True)
def calc_gap_b_sub(XSM, timeISO, rCurrents, rIonosphere, nTheta, nPhi, nR, 
                   dTheta, dPhi, dR, theta_array, phi_array, jr_array):
    """ Subroutine for calc_gap_b that allows numba accelleration.  It calculates 
    the total B field at point XSM
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        timeISO = ISO time for data in RIM file
              
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents
            
        dTheta, dPhi, dR = delta Theta, Phi, and R used in numeric integrations.
            We don't really need these as inputs, we have nTheta, nPhi, and nR,
            but they are calculated in the calling routine so no need to repeat 
            the calculation here.
            
        theta_array, phi_array, jr_array = arrays defining jr at grid of theta
            and phi points.  Values taken from ionosphere data file.  Used to
            determine initial current density along field line.
            
    Outputs:
        B = total B due to field-aligned currents (in SM coordinates)
    """

    # Set up some variables used below
    x_fac = np.zeros(3)
    b_fac = np.zeros(3)
    B = np.zeros(3)
    
    # i_tot = 0
    
    # Start the loops for the Biot-Savart numerical integration. We use three 
    # loops - theta, phi and r.  theta and phi cover the inner boundary
    # where integration begins (the sphere at rIonosphere).  r integrates 
    # over each field line as the current follows the field line up to rCurrents.  
    for i in range(nTheta):
        # theta is a latitude from pi/2 -> -pi/2
        # Find theta at the middle of each integration volume element
        # Volume is from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - (i + 0.5) * dTheta
        
        # Which hemisphere, north (+) or south (-).  We need this below to 
        # get proper signs on angles below
        hemi = np.sign(theta)
         
        ##########################################################
        # To do the numerical integration, we walk up the field line 
        # that starts at rIonosphere, theta, and phi.  We will need 
        # points along a dipole field line to map the current toward  
        # rCurrents.  To do so, we consider r (distance from earth
        # center to field line) as a function of theta0 (latitude to point
        # on the field line measured in SM cooridnates) 
        #
        # See Willis and Young, 1987, Geophys. J.R. Astr. Soc. 89, 1011-1022
        # Equation for the field lines of an axisymmetric magnetic dipole
        #
        #          r = r1*cos(theta0)**2  
        #
        # where r1 is radius to field line at equator. Note, sin became cos 
        # because we use latitude.  We will step up the field line in 
        # nR steps.  And we will use the above equation to determine 
        # x,y,z of new point on field line by varying theta0
        # 
        # We also will need the length of each of the segments along the 
        # field line.
        #
        # See Chapman and Sugiura, Journal of Geophys. Research, Vol 61, 
        # No. 3, September 1956
        #
        # ds, the arc length along dipole field line, is given by
        #
        #      ds = r1 * (1 + 3*sin(theta0)^2)^(1/2) * cos(theta0) * dtheta0
        #
        # where r1 is radius to field line at equator (theta0 = 0)
        #       theta0 is latitude to point on field line
        ##########################################################

        # Find r1 constant in Willis/Young equation
        r1 = rIonosphere / np.cos(theta)**2

        for j in range(nPhi):
            # Find phi at the middle of each integration volume element
            # Volume is from phi - dPhi/2 to phi + dPhi/2
            phi = (j + 0.5) * dPhi

            # Look for nearest neighbor (d2 min) to get jr for a point at 
            # theta, phi. We assume this is the current density at rIonosphere.
            # Note, must transform theta from latitude to elevation with 0 at 
            # north pole, pi at south pole
            d2 = (theta_array - np.pi/2 + theta)**2 + (phi_array - phi)**2
            jr = jr_array[d2.argmin()]
            
            ################################################################
            # Below we find the Field Aligned Current (FAC).  We can safely
            # ignore the current perpendicular to the magnetic field.  Lotko
            # shows j_perp = 0 and j_parallel/B_0 = constant in the ionosphere
            # and low-altitude magnetosphere
            #
            # See Lotko, 2004, J. Atmo. Solar-Terrestrial Phys., 66, 1443–1456
            ################################################################
            
            # Do integration along field line.
            for k in range(nR):
                # Find r at the middle of each integration volume element
                # Volume is from r - dR/2 to r + dR/2
                r = rIonosphere + (k + 0.5)*dR 
                
                # Find dTheta0 by looking at theta0 at top and bottom of 
                # integration volume
                rt = r + 0.5*dR
                rb = r - 0.5*dR
                
                # Check top and bottom of integration volume to see whether
                # the field line flows through it.  As appropriate adjust limits
                
                # If rb > r1, volume element is above the field line, so no contribution
                if rb <= r1: 
                    theta0b = hemi * np.arccos( np.sqrt(rb/r1) )
  
                    # if rt > r1, rt is above the field line, so limit theta0
                    # to equator 
                    if rt <= r1:
                        theta0t = hemi * np.arccos( np.sqrt(rt/r1) )
                    else:
                        rt = r1
                        theta0t = 0
                        # Put r in the middle of rt and rb
                        r = (rt + rb)/2.
                    
                    # Now that we know whether r changed, get theta0
                    theta0 = hemi * np.arccos( np.sqrt(r/r1) )
                      
                    # Get dTheta0
                    dTheta0 = np.abs( theta0b - theta0t )
                                   
                    # Find neg. inclination of B field, used to adjust current density
                    # j parallel = - jr / sin(I) for axisymmetric dipole field
                    I = np.arctan2( 2. * np.sin(theta), np.cos(theta) )
                                        
                    # FAC follows B dipole field, see Willis and Young above.
                    # Above, we used that equation to determine r to midpoint
                    # of field line segment.  Convert to cartesian.
                    x_fac[0] = r * np.cos(phi) * np.cos(theta0)
                    x_fac[1] = r * np.sin(phi) * np.cos(theta0)
                    x_fac[2] = r * np.sin(theta0)
                                        
                    # Calculate earth's magnetic field in cartesian coordinates using a 
                    # simple dipole model
                    #
                    # https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
                    #
                    # As noted in Lotko, magnetic perturbations in the ionosphere 
                    # and the low-altitude magnetosphere are much smaller than 
                    # the geomagnetic field B0.  So we can use the simple dipole field.
                    B0 = 3.12e+4 # Changed to nT units             
                    b_fac[0] = - 3 * B0 * x_fac[0] * x_fac[2] / r**5
                    b_fac[1] = - 3 * B0 * x_fac[1] * x_fac[2] / r**5
                    b_fac[2] = - B0 * ( 3 * x_fac[2]**2 - r**2 ) / r**5
                    b_fac_hat = b_fac / np.linalg.norm(b_fac)

                    ##########################################################
                    # We use that i_fac = j_fac * dSurface is constant.
                    #
                    # Lotko shows j/B_0 is a constant.  So j increases as we approach
                    # earth and the B field increases in strength.
                    #
                    #         j1 / B_01 = j2 / B_02 
                    #                   => j2 = j1 B_02 / B_01
                    #
                    # Conversely, B_0 is proportional to field line density.
                    #
                    #         B_0 <=> # of field lines / Area
                    #
                    # Since dSurface has one field line through it...
                    # 
                    #         B_01 * dSurface1 = B_02 * dSurface2 <=> 1
                    #                   => dSurface2 = dSurface1 * B_01 / B_02
                    #
                    # Together, we get:
                    #
                    #       j2 * dSurface2 = j1 * B_02 / B_01 * dSurface1 * B_01 / B_02
                    #                      = j1 * dSurface1
                    ##########################################################

                    # Since current, i_fac, is constant over a field line, we 
                    # find it once when k = 0. Aka, when the field line starts
                    if k == 0:
                        jfactor = - jr/np.sin(I) 
                        i_fac = jfactor * np.cos(theta) * rIonosphere**2 * dTheta * dPhi
                        # i_tot = i_tot - i_fac * np.sin(I) * 6371000.**2 * 10**-6
              
                    # Get length of field line element, see Chapman refereence abvoe
                    ds = r1 * (1 + 3*np.sin(theta0)**2)**(1/2) * np.cos(theta0) * dTheta0
                   
                    ##########################################################
                    # Below we calculate the delta B for each field line segment using
                    # the Biot-Savart Law.  We want the final result to be in nT.
                    # dB = mu0/(4pi) (j x r)/r^3 dV
                    #    = (4pi 10^(-7) [H/m])/(4pi) (10^(-6) [A/m^2]) [Re] [Re^3] / [Re^3]
                    # where the fact that J is in microamps/m^2 and distances are in Re
                    # is in the BATSRUS documentation.   We take Re = 6371 km = 6.371 10^6 m
                    # dB = 6.371 10^(-7) [H/m][A/m^2][m]
                    #    = 6.371 10^(-7) [H] [A/m^2]
                    #    = 6.371 10^(-7) [kg m^2/(s^2 A^2)] [A/m^2]
                    #    = 6.371 10^(-7) [kg / s^2 / A]
                    #    = 6.371 10^(-7) [T]
                    #    = 6.371 10^2 [nT]
                    #    = 637.1 [nT] with distances in Re, j in microamps/m^2
                    ##########################################################
                    
                    # Determine range to X where magnetic field is measured
                    r_fac = XSM - x_fac
                    r_fac_mag = np.linalg.norm( r_fac )

                    # In Biot-Savart, remember i_fac is in b_fac_hat direction
                    B[:] = B[:] + 637.1 * i_fac * np.cross( b_fac_hat, r_fac ) * ds / r_fac_mag**3
    
    # print('i: ', i_tot)
    return B

def calc_gap_b(XSM, filepath, timeISO, rCurrents, rIonosphere, nTheta, nPhi, nR):
    """Process data in RIM or IDL file to calculate the delta B at point XSM as
    determined by the field-aligned currents between the radius rIonosphere
    and rCurrents.  Biot-Savart Law is used for calculation.  We will integrate
    across all currents flowing through the sphere at range rIonosphere from earth
    origin.
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        filepath = path to RIM file
              
        timeISO = ISO time for data in RIM file
              
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents
            
    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(f'Calculate gap dB... {os.path.basename(filepath)} {nTheta} {nPhi} {nR}')

    ############################################################
    # Earth dipole field is in SM coordinates, and RIM is in SM
    # cordinates, so we do everything in SM coordinates
    ############################################################

    # Determine the size of the steps in theta, phi, r for numerical integration
    # over spherical shell between rIonosphere and rCurrents
    dTheta = np.pi    / nTheta
    dPhi   = 2.*np.pi / nPhi
    dR     = (rCurrents - rIonosphere) / nR
    
    base_ext = os.path.splitext( filepath )
    if base_ext[1] == '.idl':
        # If its an idl file, use spacepy Iono to read
        ionodata = Iono( filepath )

        # Make sure arrays have same dimensions
        assert ionodata['n_theta'].shape == ionodata['n_psi'].shape
        assert ionodata['n_theta'].shape == ionodata['n_jr'].shape
        assert ionodata['s_theta'].shape == ionodata['s_psi'].shape
        assert ionodata['s_theta'].shape == ionodata['s_jr'].shape

        # Get north and south hemisphere data  
        # Note, psi is the angle phi
        n_theta = ionodata['n_theta'].reshape( -1 ) 
        n_psi   = ionodata['n_psi'].reshape( -1 ) 
        n_jr    = ionodata['n_jr'].reshape( -1 ) 
        s_theta = ionodata['s_theta'].reshape( -1 )
        s_psi   = ionodata['s_psi'].reshape( -1 )
        s_jr    = ionodata['s_jr'].reshape( -1 )

        # Combine north and south hemisphere data and setup interpolator for 
        # finding jr at point theta, phi on 2D surface from IDL file.  NOTE, 
        # must transform input data from degrees to radians.
        # Note: theta is 0 -> pi
        theta_array = np.concatenate( [n_theta, s_theta], axis = 0 ) * np.pi/180
        phi_array = np.concatenate( [n_psi, s_psi], axis = 0 ) * np.pi/180
        jr_array = np.concatenate( [n_jr, s_jr], axis = 0 )
       
    else:
        # Read RIM file
        data_arr, var_dict, units = swmfio.read_rim(filepath)
        assert(data_arr.shape[0] != 0)
    
        # Setup interpolator for finding jr at point theta, phi on 2D surface from 
        # RIM file.  NOTE, must transform input data from degrees to radians
        theta_array = data_arr[var_dict['Theta']][:] * np.pi/180
        phi_array = data_arr[var_dict['Psi']][:] * np.pi/180
        jr_array = data_arr[var_dict['JR']][:]

    B = calc_gap_b_sub(XSM, timeISO, rCurrents, rIonosphere, nTheta, nPhi, nR,
                        dTheta, dPhi, dR, theta_array, phi_array, jr_array)
    
    Bn, Be, Bd = get_NED_components( B, XSM )
    
    return Bn, Be, Bd, B[0], B[1], B[2]      
    
@jit(nopython=True)
def integrate_r_sub( nR, dR, rIonosphere, r1, hemi, theta, dTheta, phi, dPhi, 
                         jr, XSM):
    """ Subroutine for calc_gap_b_rim that allows numba accelleration.  It calculates 
    total B field at point XSM using data from a RIM file.  Data from a RIM file,
    either as a .tec or .cdf, has a known structure (from reverse engineering) 
    that we utilize in the integration
    
    Inputs:
        nR = number of points to be examined in the numerical integration. nR 
            points in spherical grid between rIonosphere and rCurrents.  nTheta
            and nPhi are known from the structure of the RIM file.
           
        dR = delta R used in numeric integrations.  We don't really need this
            input, we have nR, but dR are calculated in the calling routine so 
            no need to repeat the calculation here.
              
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        hemi = north (1) or south (-1) hemisphere
        
        theta = latitude pi/2 -> -pi/2
        
        dTheta = spacing between points on grid
        
        phi = longitude 0 -> 2 pi
        
        dPhi = spacing between points on grid
        
        jr = radial current density at theta, phi, rIonosphere
        
        XSM = SM (cartesian) position where magnetic field will be measured.
            Dipole data is in SM coordinates
        
    Outputs:
        B = total B due to field-aligned currents (in SM coordinates)
    """
    
    ##########################################################
    # To do the numerical integration, we walk  up the field line 
    # that starts at rIonosphere, theta, and phi.  We will need 
    # points along a dipole field line to map the current toward  
    # rCurrents.  To do so, we consider r (distance from earth
    # center to field line) as a function of theta0 (latitude to point
    # on the field line measured in SM cooridnates) 
    #
    # See Willis and Young, 1987, Geophys. J.R. Astr. Soc. 89, 1011-1022
    # Equation for the field lines of an axisymmetric magnetic dipole
    #
    #          r = r1*cos(theta0)**2  
    #
    # where r1 is radius to field line at equator. Note, sin became cos 
    # because we use latitude.  We will step up the field line in 
    # nR steps.  And we will use the above equation to determine 
    # x,y,z of new point on field line by varying theta0
    # 
    # We also will need the length of each of the segments along the 
    # field line.
    #
    # See Chapman and Sugiura, Journal of Geophys. Research, Vol 61, 
    # No. 3, September 1956
    #
    # ds, the arc length along dipole field line, is given by
    #
    #      ds = r1 * (1 + 3*sin(theta0)^2)^(1/2) * cos(theta0) * dtheta0
    #
    # where r1 is radius to field line at equator (theta0 = 0)
    #       theta0 is latitude to point on field line
    ##########################################################
    
    ################################################################
    # Below we find the Field Aligned Current (FAC).  We can safely
    # ignore the current perpendicular to the magnetic field.  Lotko
    # shows j_perp = 0 and j_parallel/B_0 = constant in the ionosphere
    # and low-altitude magnetosphere
    #
    # See Lotko, 2004, J. Atmo. Solar-Terrestrial Phys., 66, 1443–1456
    ################################################################
    
    # Set up some variables used below
    x_fac = np.zeros(3)
    b_fac = np.zeros(3)
    B = np.zeros(3)
      
    # i_tot = 0

    # Do integration along field line.
    for k in range(nR):
        # Find r at the middle of each integration volume element
        # Volume is from r - dR/2 to r + dR/2
        r = rIonosphere + (k + 0.5)*dR 
        
        # Find dTheta0 by looking at theta0 at top and bottom of 
        # integration volume
        rt = r + 0.5*dR
        rb = r - 0.5*dR
        
        # Check top and bottom of integration volume to see whether
        # the field line flows through it.  As appropriate adjust limits
        
        # If rb > r1, volume element is above the field line, so no contribution
        if rb <= r1: 
            theta0b = hemi * np.arccos( np.sqrt(rb/r1) )
  
            # if rt > r1, rt is above the field line, so limit theta0
            # to equator 
            if rt <= r1:
                theta0t = hemi * np.arccos( np.sqrt(rt/r1) )
            else:
                rt = r1
                theta0t = 0
                # Put r in the middle of rt and rb
                r = (rt + rb)/2.
            
            # Now that we know whether r changed, get theta0
            theta0 = hemi * np.arccos( np.sqrt(r/r1) )
              
            # Get dTheta0
            dTheta0 = np.abs( theta0b - theta0t )
                           
            # Find neg. inclination of B field, used to adjust current density
            # j parallel = - jr / sin(I) for axisymmetric dipole field
            I = np.arctan2( 2. * np.sin(theta), np.cos(theta) )
                                
            # FAC follows B dipole field, see Willis and Young above.
            # Above, we used that equation to determine r to midpoint
            # of field line segment.  Convert to cartesian.
            x_fac[0] = r * np.cos(phi) * np.cos(theta0)
            x_fac[1] = r * np.sin(phi) * np.cos(theta0)
            x_fac[2] = r * np.sin(theta0)
                                
            # Calculate earth's magnetic field in cartesian coordinates using a 
            # simple dipole model
            #
            # https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
            #
            # As noted in Lotko, magnetic perturbations in the ionosphere 
            # and the low-altitude magnetosphere are much smaller than 
            # the geomagnetic field B0.  So we can use the simple dipole field.
            B0 = 3.12e+4 # Changed to nT units             
            b_fac[0] = - 3 * B0 * x_fac[0] * x_fac[2] / r**5
            b_fac[1] = - 3 * B0 * x_fac[1] * x_fac[2] / r**5
            b_fac[2] = - B0 * ( 3 * x_fac[2]**2 - r**2 ) / r**5
            b_fac_hat = b_fac / np.linalg.norm(b_fac)

            ##########################################################
            # We use that i_fac = j_fac * dSurface is constant.
            #
            # Lotko shows j/B_0 is a constant.  So j increases as we approach
            # earth and the B field increases in strength.
            #
            #         j1 / B_01 = j2 / B_02 
            #                   => j2 = j1 B_02 / B_01
            #
            # Conversely, B_0 is proportional to field line density.
            #
            #         B_0 <=> # of field lines / Area
            #
            # Since dSurface has one field line through it...
            # 
            #         B_01 * dSurface1 = B_02 * dSurface2 <=> 1
            #                   => dSurface2 = dSurface1 * B_01 / B_02
            #
            # Together, we get:
            #
            #       j2 * dSurface2 = j1 * B_02 / B_01 * dSurface1 * B_01 / B_02
            #                      = j1 * dSurface1
            ##########################################################

            # Since current, i_fac, is constant over a field line, we 
            # find it once when k = 0. Aka, when the field line starts
            if k == 0:
                jfactor = - jr/np.sin(I) 
                i_fac = jfactor * np.cos(theta) * rIonosphere**2 * dTheta * dPhi
                # i_tot = i_tot - i_fac * np.sin(I) * 6371000.**2 * 10**-6
      
            # Get length of field line element, see Chapman refereence abvoe
            ds = r1 * (1 + 3*np.sin(theta0)**2)**(1/2) * np.cos(theta0) * dTheta0
           
            ##########################################################
            # Below we calculate the delta B for each field line segment using
            # the Biot-Savart Law.  We want the final result to be in nT.
            # dB = mu0/(4pi) (j x r)/r^3 dV
            #    = (4pi 10^(-7) [H/m])/(4pi) (10^(-6) [A/m^2]) [Re] [Re^3] / [Re^3]
            # where the fact that J is in microamps/m^2 and distances are in Re
            # is in the BATSRUS documentation.   We take Re = 6371 km = 6.371 10^6 m
            # dB = 6.371 10^(-7) [H/m][A/m^2][m]
            #    = 6.371 10^(-7) [H] [A/m^2]
            #    = 6.371 10^(-7) [kg m^2/(s^2 A^2)] [A/m^2]
            #    = 6.371 10^(-7) [kg / s^2 / A]
            #    = 6.371 10^(-7) [T]
            #    = 6.371 10^2 [nT]
            #    = 637.1 [nT] with distances in Re, j in microamps/m^2
            ##########################################################
            
            # Determine range to X where magnetic field is measured
            r_fac = XSM - x_fac
            r_fac_mag = np.linalg.norm( r_fac )

            # In Biot-Savart, remember i_fac is in b_fac_hat direction
            B[:] = B[:] + 637.1 * i_fac * np.cross( b_fac_hat, r_fac ) * ds / r_fac_mag**3
           
    # return B, i_tot
    return B

@jit(nopython=True)
def calc_gap_b_rim_sub(XSM, timeISO, rCurrents, rIonosphere, nR, dR, 
                       theta_array, phi_array, jr_array):
    """ Subroutine for calc_gap_b_rim that allows numba accelleration.  It calculates 
    total B field at point XSM using data from a RIM file.  Data from a RIM file,
    either as a .tec or .cdf, has a known structure (from reverse engineering) 
    that we utilize in the integration
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        timeISO = ISO time for data in RIM file
              
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nR = number of points to be examined in the numerical integration. nR 
            points in spherical grid between rIonosphere and rCurrents.  nTheta
            and nPhi are known from the structure of the RIM file.
            
        dR = delta R used in numeric integrations.  We don't really need this
            input, we have nR, but dR are calculated in the calling routine so 
            no need to repeat the calculation here.
            
        theta_array, phi_array, jr_array = arrays defining jr at grid of theta
            and phi points.  Values taken from ionosphere RIM file.  Used to
            determine initial current density along field line.
            
    Outputs:
        B = total B due to field-aligned currents (in SM coordinates)
    """

    # algorithm is based on reverse-engineering of RIM format, so check to
    # see if this is the right-size for expected data.
    #
    # 32222 = 2 (poles, 0 and 180 lat) + 179 (1 deg lat 1, 2, ... 179) x 180 (2 
    # deg long 0, 2, ... 358) 
    assert( len(jr_array) == 32222 )
    # Verify first two entries are for North and South poles
    assert( np.abs(theta_array[0]) < 10**-5 ) # 
    assert( np.abs(theta_array[1] - np.pi) < 10**-5 )
    
 
    # Set up some variables used below
    # x_fac = np.zeros(3)
    # b_fac = np.zeros(3)
    B = np.zeros(3)
    Bsub = np.zeros(3)
    
    # i_tot = 0
    # i_sub = 0

    # Start the loops for the Biot-Savart numerical integration. We use three 
    # loops - theta, phi and r.  theta and phi cover the inner boundary
    # where integration begins (the sphere at rIonosphere).  r integrates 
    # over each field line as the current follows the field line up to rCurrents.  
    for i in range(2,32222):
        # theta is a latitude from pi/2 -> -pi/2
        # Find theta at the middle of each integration volume element
        # Volume is from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - theta_array[i]
        dTheta = 1. * np.pi/180
        
        # Handle special case of equator (theta = 0), where the dipole field is 
        # tangential to the earth, and so there is no FAC.  i.e., we skip the  
        # magnetic equator
        if np.abs(theta) > 10**-5:

            # Which hemisphere, north (+) or south (-).  We need this below to 
            # get proper signs on angles below
            hemi = np.sign(theta)
             
            # Find r1 constant in Willis/Young equation
            r1 = rIonosphere / np.cos(theta)**2
    
            # Find phi at the middle of each integration volume element
            # Volume is from phi - dPhi/2 to phi + dPhi/2
            phi = phi_array[i]
            dPhi = 2. * np.pi/180
    
            # Look for nearest neighbor (d2 min) to get jr for a point at 
            # theta, phi. We assume this is the current density at rIonosphere.
            # Note, must transform theta from latitude to elevation with 0 at 
            # north pole, pi at south pole
            jr = jr_array[i]
            
            # Bsub, i_sub = integrate_r_sub( nR, dR, rIonosphere, r1, hemi, theta, 
            #                            dTheta, phi, dPhi, jr, XSM)
            Bsub = integrate_r_sub( nR, dR, rIonosphere, r1, hemi, theta, 
                                       dTheta, phi, dPhi, jr, XSM)
            B[:] = B[:] + Bsub[:]
            # i_tot = i_tot + i_sub
            
    # print('i1: ', i_tot)
    # print('B1: ', B)
    
    # Integrate over the poles.  Above we integrated over latitudes in one degree
    # increments, plus or minus a half a degree.  This leaves end caps at the poles
    # that have not been integrated.  We set theta = 0.25 and 179.75 deg, and 
    # vary phi in 2 degree increments to integrate over the end caps.
    
    # First two jr_array entries are North and South poles
    jrN = jr_array[0]
    jrS = jr_array[1]

    # Get some constants: theta fixed, phi varies.  As above,
    # volume is from theta - dTheta/2 to theta + dTheta/2 
    # and from phi - dPhi/2 to phi + dPhi/2
    thetaN = np.pi/2 - 0.25 * np.pi/180
    thetaS = np.pi/2 - 179.75 * np.pi/180
    dTheta = 0.5 * np.pi/180 

    for i in range(0,360,2):
        
        phi = i * np.pi/180
        
        # North pole
        hemiN = 1       
        r1N = rIonosphere / np.cos(thetaN)**2
        B[:] = B[:] + integrate_r_sub( nR, dR, rIonosphere, r1N, hemiN, thetaN, 
                                        dTheta, phi, dPhi, jrN, XSM)
        # Bsub, i_sub = integrate_r_sub( nR, dR, rIonosphere, r1N, hemiN, thetaN, 
        #                                 dTheta, phi, dPhi, jrN, XSM)
       
        B[:] = B[:] + Bsub[:]
        # i_tot = i_tot + i_sub
        
        # South pole
        hemiS = -1       
        r1S = rIonosphere / np.cos(thetaS)**2
        B[:] = B[:] + integrate_r_sub( nR, dR, rIonosphere, r1S, hemiS, thetaS, 
                                        dTheta, phi, dPhi, jrS, XSM)
        # Bsub, i_sub = integrate_r_sub( nR, dR, rIonosphere, r1S, hemiS, thetaS, 
        #                                 dTheta, phi, dPhi, jrS, XSM)
        B[:] = B[:] + Bsub[:]
        # i_tot = i_tot + i_sub

    # print('i2: ', i_tot)
    # print('B2: ', B)
    
    return B
  
def calc_gap_b_rim(XSM, filepath, timeISO, rCurrents, rIonosphere, nR):
    """Process data in RIM file to calculate the delta B at point XSM as
    determined by the field-aligned currents between the radius rIonosphere
    and rCurrents.  Biot-Savart Law is used for calculation.  We will integrate
    across all currents flowing through the sphere at range rIonosphere from earth
    origin. This routine is similar to calc_gap_b which accepts both RIM and IDL
    files. However,this routine accepts only RIM files, and uses a reverse-engineered
    understanding of the RIM format to numerically integrate the data.  
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        filepath = path to RIM file
              
        timeISO = ISO time for data in RIM file
              
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

    logging.info(f'Calculate gap dB... {os.path.basename(filepath)} {nR}')

    ############################################################
    # Earth dipole field is in SM coordinates, and RIM is in SM
    # coordinates, so we do everything in SM coordinates
    ############################################################

    # Determine the size of the steps in r for numerical integration
    # over spherical shell between rIonosphere and rCurrents
    dR = (rCurrents - rIonosphere) / nR
    
    # Read RIM file
    data_arr, var_dict, units = swmfio.read_rim(filepath)
    assert(data_arr.shape[0] != 0)

    # Setup interpolator for finding jr at point theta, phi on 2D surface from 
    # RIM file.  NOTE, must transform input data from degrees to radians
    theta_array = data_arr[var_dict['Theta']][:] * np.pi/180
    phi_array = data_arr[var_dict['Psi']][:] * np.pi/180
    jr_array = data_arr[var_dict['JR']][:]

    B = calc_gap_b_rim_sub(XSM, timeISO, rCurrents, rIonosphere, nR, dR, 
                            theta_array, phi_array, jr_array)
    
    Bn, Be, Bd = get_NED_components( B, XSM )
    
    return Bn, Be, Bd, B[0], B[1], B[2]      
    
# Example info.  Info is used below in call to loop_ms_b
# info = {
#         "model": "SWMF",
#         "run_name": "SWPC_SWMF_052811_2",
#         "rCurrents": 4.0,
#         "rIonosphere": 1.01725,
#         "file_type": "cdf",
#         "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
#         "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
#         "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
#         "deltaB_files": {
#             "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
#         }
# }

def loop_gap_b(info, point, reduce, nTheta=180, nPhi=180, nR=800, useRIM=True, 
               deltahr=None, maxcores=20):
    """Use Biot-Savart in calc_gap_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Biot-Savart caclculation 
    uses ionospheric current density as defined in BATSRUS files

    Inputs:
        info = information on BATSRUS data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
            
        reduce = Do we skip files to save time. If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        useRIM = Boolean, if False use calc_gap_b, if True use calc_gap_b_rim
            The difference is calc_gap_b makes no assumptions about the RIM file
            while calc_gap_b assumes a structure to the RIM file based on 
            reverse engineering.

        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.
            
        maxcores = for parallel processing, the maximum number of cores to use
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_gap( i, times, deltahr, XGEO, info, nTheta, nPhi, nR, useRIM ):
        time = times[i]
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['ionosphere'][time]
        base = os.path.basename(filepath)

        logging.info(f'Calculate gap dB for... {base}')
        
        # We need the ISO time to update the magnetometer position
        # Record time for plots
        if deltahr is None:
            h = time[3]
            m = time[4]
            Btime = h + m/60
            timeISO = date_timeISO( time )
        else:
            dtime = datetime(*time) + timedelta(hours=deltahr)
            timeISO = dtime.isoformat()
            h = dtime.hour
            m = dtime.minute
            Btime = h + m/60
        
        # Get the magnetometer position, X, in SM coordinates for compatibility with
        # RIM data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XSM = XGEO.convert( 'SM', 'car' )
        X = XSM.data[0]
            
        # Use Biot-Savart to calculate magnetic field, B, at magnetometer position
        # XSM.  Store the results and the time
        if useRIM:
            Bn, Be, Bd, Bx, By, Bz = calc_gap_b_rim(X, filepath, timeISO, \
                                    info['rCurrents'], info['rIonosphere'], nR)
        else:
            Bn, Be, Bd, Bx, By, Bz = calc_gap_b(X, filepath, timeISO, \
                                    info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)
                
        return Bn, Be, Bd, Bx, By, Bz, Btime

    # Verify input parameters
    assert isinstance(point, str)

    # Make sure delta_hr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Warn user if nTheta and nPhi are ignored
    if useRIM:
        logging.info("nTheta and nPhi values ignored when useRIM is True")
    else:
        logging.warning("Warning: Depreciated mode, useRIM=True is recommended")

    # Get times for RIM files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types
    times = list(info['files']['ionosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    # from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)
    
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_gap)( p, times, deltahr, \
                                   XGEO, info, nTheta, nPhi, nR, useRIM )
                                   for p in range(len(times)))
        
        Bn, Be, Bd, Bx, By, Bz, Btimes = zip(*results)

    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bn = np.zeros(n)
        Be = np.zeros(n)
        Bd = np.zeros(n)
        Bx = np.zeros(n)
        By = np.zeros(n)
        Bz = np.zeros(n)
        
        Btimes = [None] * n

        for p in range(len(times)):
            Bn[p], Be[p], Bd[p], Bx[p], By[p], Bz[p], Btimes[p] = \
                wrap_gap( p, times, deltahr, XGEO, info, nTheta, nPhi, nR, useRIM )

    # Create dataframe from results and save to disk
    if deltahr is None:
        dtimes = [datetime(*time) for time in times]
    else:
        dtimes = [datetime(*time) + timedelta(hours=deltahr) for time in times]
        
    dtimes_m = [dtime.month for dtime in dtimes]
    dtimes_d = [dtime.day for dtime in dtimes]
    dtimes_hh = [dtime.hour for dtime in dtimes]
    dtimes_mm = [dtime.minute for dtime in dtimes]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bn': Bn, 'Be': Be, 'Bd': Bd,
                        'Bx': Bx, 'By': By, 'Bz': Bz, 
                        r'Time (hr)': Btimes, r'Datetime': dtimes,
                        r'Month': dtimes_m, r'Day': dtimes_d,
                        r'Hour': dtimes_hh, r'Minute': dtimes_mm}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_bs_gap-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )


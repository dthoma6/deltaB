#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:10:35 2023

@author: Dean Thomas
"""

import logging
import numpy as np
import swmfio
import pandas as pd
from datetime import datetime
from scipy.interpolate import NearestNDInterpolator
import os.path

# from deltaB.coordinates import get_transform_matrix
# from deltaB.util import  create_directory
from deltaB.util import create_directory, get_NED_components
# from deltaB.coordinates import get_NED_vector_components
# from deltaB.util import get_NED_components

# Setup logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

MYMETHOD = True
DOTEST = False

def calc_gap_b(XSM, filepath, rCurrents, rIonosphere, nTheta, nPhi, nR):
    """Process data in RIM file to calculate the delta B at point XGSM as
    determined by the field-aligned currents between the radius rCurrents
    and the rIonosphere.  Biot-Savart Law is used for calculation.  We will integrate
    across all currents flowing through the sphere at range rCurrents from earth
    origin.
    
    Inputs:
        XSM = SM (cartesian) position where magnetic field will be measured.
        Dipole data is in SM coordinates
        
        filepath = path to RIM file
              
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nTheta, nPhi, nR = number of points to be examined in the 
            numerical integration. nTheta, nPhi, nR points in spherical grid
            between rIonosphere and rCurrents
            
    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X

        bSM = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(f'Calculate FAC dB... {os.path.basename(filepath)}')

    # Set up some variables used below
    x_fac = np.empty(3)
    b = np.empty(3)
    b_fac = np.empty(3)

    ############################################################
    # Earth dipole field is in SM coordinates, and RIM is in SM
    # cordinates, so we do everything in SM coordinates
    ############################################################

    # Determine the size of the steps in theta, phi, r for numerical integration
    # over spherical shell between rIonosphere and rCurrents
    dTheta = np.pi    / nTheta
    dPhi   = 2.*np.pi / nPhi
    dR     = (rCurrents - rIonosphere) / nR
    
    if DOTEST:
        def jtp_interpolate( theta, phi ):
            # jr for a line current flowing from north to south pole.
            # jr independent of phi
            # theta in radian 0 -> pi
            
            num = 1
            jr = 10**10 * np.cos(theta) / (np.pi * (6.371*10**6 * np.sin(num*dTheta))**2)           
        
            # North pole
            if theta < num*dTheta:
                # 10 kA -> 10^4 A -> 10^10 muA
                # over pi R^2 where R is a small disk at top of earth 
                # inside elevation theta -> Re sin(theta)
                # convert to jr using dot product -> cos(theta)
                # print( 'N', theta )
                return jr
            
            # South pole
            if theta > (np.pi - num*dTheta):
                # print('S', theta)
                return -jr
            
            # print('Neither', theta)
            return 0
    else:
        # Read RIM file
        # swmfio.logger.setLevel(logging.INFO)
        data_arr, var_dict, units = swmfio.read_rim(filepath)
        assert(data_arr.shape[0] != 0)
    
        # Setup interpolator for finding jr at point theta, phi on 2D surface from 
        # RIM file.  NOTE, must transform input data from degrees to radians
        theta_array = data_arr[var_dict['Theta']][:] * np.pi/180
        phi_array = data_arr[var_dict['Psi']][:] * np.pi/180
        jr_array = data_arr[var_dict['JR']][:]
        jtp_interpolate = NearestNDInterpolator( list(zip( theta_array, phi_array )), jr_array )

    # Start the loops for the Biot-Savart numerical integration. We use three 
    # loops - theta, phi and r.  theta and phi cover the inner boundary
    # where integration begins (the sphere at rIonosphere).  r integrates 
    # over each field line as the current follows field lines up to rCurrents.  
    for i in range(nTheta):
        # theta is a latitude from pi/2 -> -pi/2
        # Find theta at the middle of each integration volume element
        # Volume is from theta - dTheta/2 to theta + dTheta/2
        theta = np.pi/2 - (i + 0.5) * dTheta
        
        # Which hemisphere, north (+) or south (-).  We need this below to 
        # get proper signs on angles below
        hemi = np.sign(theta)
         
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

        # Find r1 constant in Willis/Young equation
        r1 = rIonosphere / np.cos(theta)**2

        for j in range(nPhi):
            # Find phi at the middle of each integration volume element
            # Volume is from phi - dPhi/2 to phi + dPhi/2
            phi = (j + 0.5) * dPhi

            # Use interpolator to get jr for a point at theta, phi. We assume
            # this is the current density at rIonosphere.
            # Note, must transform theta from latitude to elevation with 0 at 
            # north pole, pi at south pole
            jr = jtp_interpolate( np.pi/2 - theta, phi )
  
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
                                   
                    # Find dV for integration volume element
                    dV = np.cos(theta0) * r**2 * dTheta0 * dPhi * np.abs( rb - rt )
                    # dV = np.cos(theta0) * r**3 * dTheta0 * dPhi * dTheta
                     
                    # Find neg. inclination of B field, used to adjust current density
                    # j parallel = - jr / sin(I) for axisymmetric dipole field
                    I = np.arctan2( 2. * np.sin(theta), np.cos(theta) )
                                        
                    # FAC follows B dipole field, see Willis and Young above.
                    # Use that equation to determine x,y,z of new point on field line
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
                    # For MYMETHOD, we use that i_fac = j_fac * dSurface is constant.
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

                    # Find j_fac, the FAC 
                    if MYMETHOD:
                        if k == 0:
                            jfactor = - jr/np.sin(I) 
                            i_fac = jfactor * b_fac_hat * np.cos(theta) * r**2 * dTheta * dPhi
                    
                        # jfactor = - jr/np.sin(I) * rIonosphere**3 / r**3 * \
                        #     np.sqrt( 3.*np.sin(theta0)**2 + 1.) / np.sqrt( 3.*np.sin(theta)**2 + 1.)  
                        # i_fac = jfactor * b_fac_hat * np.cos(theta0) * r**2 * dTheta * dPhi
                        ds = r1 * (1 + 3*np.sin(theta0)**2)**(1/2) * np.cos(theta0) * dTheta0
                    else:
                        jfactor = - jr/np.sin(I) * rIonosphere**3 / r**3 * \
                            np.sqrt( 3.*np.sin(theta0)**2 + 1.) / np.sqrt( 3.*np.sin(theta)**2 + 1.)                     
                        j_fac = jfactor * b_fac_hat
                   
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

                    if MYMETHOD:
                        b[:] = b[:] + 637.1 * np.cross( i_fac, r_fac ) * ds / r_fac_mag**3
                    else:    
                        # b[:] = b[:] + 637.1 * np.cross( j_fac, r_fac ) * dV / r_fac_mag**3
                        b[:] = b[:] + 637.1 * np.cross( j_fac, r_fac ) * dV / r_fac_mag**3
    
    # Bn2, Be2, Bd2 = get_NED_vector_components( b.reshape(1,3), X.reshape(1,3) ).ravel()
    Bn2, Be2, Bd2 = get_NED_components( b, XSM )
    
    return Bn2, Be2, Bd2, b[0], b[1], b[2]   

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

def loop_gap_b(info, point, reduce, nTheta, nPhi, nR):
    """Use Biot-Savart in calc_ms_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Biot-Savart caclculation 
    uses gap region current density as defined in RIM files

    Inputs:
        info = information on RIM data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
            
        reduce = Boolean, do we skip files to save time
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    import os.path
    
    assert isinstance(point, str)

    # Get a list of RIM files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types
    from magnetopost import util as util
    util.setup(info)
    
    times = list(info['files']['ionosphere'].keys())
    if reduce:
        times = times[0:len(times):60]
    n = len(times)

    # Prepare storage of variables
    Bn = np.zeros(n)
    Be = np.zeros(n)
    Bd = np.zeros(n)
    Bx = np.zeros(n)
    By = np.zeros(n)
    Bz = np.zeros(n)
    
    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype)
    
    # Loop through each RIM file, storing the results along the way
    for i in range(n):
        
        # We need the filepath for RIM file
        filepath = info['files']['ionosphere'][times[i]]
        
        # We need the ISO time to update the magnetometer position
        timeISO = str(times[i][0]) + '-' + str(times[i][1]).zfill(2) + '-' + str(times[i][2]).zfill(2) + 'T' + \
            str(times[i][3]).zfill(2) +':' + str(times[i][4]).zfill(2) + ':' + str(times[i][5]).zfill(2)
        
        # Get the magnetometer position, X, in SM coordinates for compatibility with
        # RIM data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XSM = XGEO.convert( 'SM', 'car' )
        X = XSM.data[0]
            
        # Use Biot-Savart to calculate magnetic field, B, at magnetometer position
        # XSM.  Store the results and the time
        Bn[i], Be[i], Bd[i], Bx[i], By[i], Bz[i] = calc_gap_b(X, filepath, \
                                info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)
    
    dtimes = [datetime(*time) for time in times]

    # Create a dataframe from the results and save it in a pickle file
    df = pd.DataFrame( data={'Bn': Bn, 'Be': Be, 'Bd': Bd,
                        'Bx': Bx, 'By': By, 'Bz': Bz}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    pklname = 'dB_bs_gap-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    
if __name__ == "__main__":
    import os

    # Prior to running this script, execute the following from the directory of
    # this script:
    #   mkdir runs && cd runs
    #   wget -r -np -nH -R "index.html*" --cut-dirs=2 http://mag.gmu.edu/git-data/ccmc/SWPC_SWMF_052811_2/
    #   cd SWPC_SWMF_052811_2; find . -name "*.cdf.gz" | xargs -J{} gunzip {}

    data_dir = '/Users/dean/Documents/GitHub/magnetopost/runs'

    # TODO: Get rCurrents from file
    info = {
            "model": "SWMF",
            "run_name": "SWPC_SWMF_052811_2",
            "rCurrents": 3.0,
            "rIonosphere": 1.01727,
            "file_type": "cdf",
            "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
            "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
            "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
            "deltaB_files": {
                "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
            }
    }

    assert DOTEST
    
    # Locations to compute B. See config.py for list of known points.
    points  = ["YKC"]
    reduce = False
    # nTheta = 180
    # nPhi = 180
    # nR = 800
    nTheta = 60
    nPhi = 60
    nR = 800

    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[points[0]]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype)
    XGEO.ticks = Ticktock(['2020-06-01T12:00:00'], 'ISO')
    XSM = XGEO.convert( 'SM', 'car' )
    X = XSM.data[0]
 
    Bn, Be, Bd, Bx, By, Bz = calc_gap_b(X, '/', info['rCurrents'], info['rIonosphere'], nTheta, nPhi, nR)
    
    print( 'Bn: ', Bn )
    print( 'Be: ', Be )
    print( 'Bd: ', Bd )
    
    XSM2 = XGEO.convert( 'SM', 'sph' )
    print( XSM2.data[0] )
    
    I=10**4 
    mu0 = 4*np.pi*10**-7
    r0 = np.cos( XSM2.data[0][1]*np.pi/180 )
    z0 = np.sin( XSM2.data[0][1]*np.pi/180 )
    z1 = info['rIonosphere']
    z2 = info['rCurrents']
    Re = 6.3712*10**6
    
    db = 10**9 * ( mu0 * I )/( 4 * np.pi * r0 * Re )*(  (z2-z0)/(np.sqrt((z2-z0)**2 + r0**2)) -\
                                          (z1-z0)/(np.sqrt((z1-z0)**2 + r0**2)))
    print('db: ', db)

    
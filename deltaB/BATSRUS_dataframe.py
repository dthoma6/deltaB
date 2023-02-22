#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 08:11:30 2022

@author: Dean Thomas
"""
import logging
from copy import deepcopy
import swmfio
from os.path import exists
import pandas as pd
import numpy as np
import deltaB.coordinates as coord
import deltaB.util as util

def convert_BATSRUS_to_dataframe(file, rCurrents):
    """Process data in BATSRUS file to create dataframe.  Biot-Savart Law used
    to calculate delta B in each BATSRUS cell.  In addition, current j and 
    magnetic field dB is determined in various coordinate systems (i.e., spherical 
    coordinates and parallel/perpendicular to B field).
    
    Inputs:
        file = path to BATSRUS file
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units.  We drop the data inside radius rCurrents
    Outputs:
        df = dataframe containing data from BATSRUS file plus additional calculated
            parameters
    """

    logging.info('Parsing BATSRUS file...')

    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(file)
    assert(batsrus != None)

    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)

    df = pd.DataFrame()

    df['x'] = batsrus.data_arr[:, var_dict['x']][:]
    df['y'] = batsrus.data_arr[:, var_dict['y']][:]
    df['z'] = batsrus.data_arr[:, var_dict['z']][:]

    df['bx'] = batsrus.data_arr[:, var_dict['bx']][:]
    df['by'] = batsrus.data_arr[:, var_dict['by']][:]
    df['bz'] = batsrus.data_arr[:, var_dict['bz']][:]

    df['jx'] = batsrus.data_arr[:, var_dict['jx']][:]
    df['jy'] = batsrus.data_arr[:, var_dict['jy']][:]
    df['jz'] = batsrus.data_arr[:, var_dict['jz']][:]

    df['ux'] = batsrus.data_arr[:, var_dict['ux']][:]
    df['uy'] = batsrus.data_arr[:, var_dict['uy']][:]
    df['uz'] = batsrus.data_arr[:, var_dict['uz']][:]

    df['p'] = batsrus.data_arr[:, var_dict['p']][:]
    df['rho'] = batsrus.data_arr[:, var_dict['rho']][:]
    df['measure'] = batsrus.data_arr[:, var_dict['measure']][:]

    # Determine magnitude of various vectors
    df['jMag'] = np.sqrt(df['jx']**2 + df['jy']**2 + df['jz']**2)
    df['uMag'] = np.sqrt(df['ux']**2 + df['uy']**2 + df['uz']**2)

    # Calculate useful quantities
    df['r0'] = ((df['x'])**2+(df['y'])**2+(df['z'])**2)**(1/2)

    # We ignore everything inside of rCurrents
    df.drop(df[df['r0'] < rCurrents].index)

    logging.info('Transforming j to spherical coordinates...')

    # Transform the currents, j, into spherical coordinates

    # Determine theta and phi of the radius vector from the origin to the
    # center of the cell
    df['theta'] = np.arccos(df['z']/df['r0'])
    df['phi'] = np.arctan2(df['y'], df['x'])

    # Use dot products with r-hat, theta-hat, and phi-hat of the radius vector
    # to determine the spherical components of the current j.
    df['jr'] = df['jx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jz'] * np.cos(df['theta'])

    df['jtheta'] = df['jx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jy'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jz'] * np.sin(df['theta'])

    df['jphi'] = - df['jx'] * np.sin(df['phi']) + df['jy'] * np.cos(df['phi'])

    logging.info('Determining j parallel and perpendicular to B...')

    # Use dot product with B/BMag to get j parallel to magnetic field
    # To get j perpendicular, use j perpendicular = j - j parallel 
    df['bMag'] = np.sqrt(df['bx']**2 + df['by']**2 + df['bz']**2)

    df['jparallelMag'] = (df['jx'] * df['bx'] + df['jy']
                          * df['by'] + df['jz'] * df['bz'])/df['bMag']

    df['jparallelx'] = df['jparallelMag'] * df['bx']/df['bMag']
    df['jparallely'] = df['jparallelMag'] * df['by']/df['bMag']
    df['jparallelz'] = df['jparallelMag'] * df['bz']/df['bMag']

    df['jperpendicularMag'] = np.sqrt(df['jMag']**2 - df['jparallelMag']**2)

    df['jperpendicularx'] = df['jx'] - df['jparallelx']
    df['jperpendiculary'] = df['jy'] - df['jparallely']
    df['jperpendicularz'] = df['jz'] - df['jparallelz']
    
    # Convert j perpendicular to spherical coordinates
    df['jperpendicularr'] = df['jperpendicularx'] * np.sin(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.sin(df['theta']) * np.sin(df['phi']) + \
        df['jperpendicularz'] * np.cos(df['theta'])

    df['jperpendiculartheta'] = df['jperpendicularx'] * np.cos(df['theta']) * np.cos(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['theta']) * np.sin(df['phi']) - \
        df['jperpendicularz'] * np.sin(df['theta'])

    df['jperpendicularphi'] = - df['jperpendicularx'] * np.sin(df['phi']) + \
        df['jperpendiculary'] * np.cos(df['phi'])

    # Divide j perpendicular into a phi piece and everything else (residual)
    df['jperpendicularphix'] = - df['jperpendicularphi'] * np.sin(df['phi'])
    df['jperpendicularphiy'] =   df['jperpendicularphi'] * np.cos(df['phi'])
    # df['jperpendicularphiz'] = 0

    df['jperpendicularphiresx'] = df['jperpendicularx'] - df['jperpendicularphix']
    df['jperpendicularphiresy'] = df['jperpendiculary'] - df['jperpendicularphiy']
    # df['jperpendicularphiresz'] = df['jperpendicularz']
    
    return df

def create_deltaB_rCurrents_dataframe(df, X):
    """Convert the dataframe with BATSRUS data to a dataframe that uses 
    Biot-Savart Law to calculate delta B in each BATSRUS cell outside of 
    rCurrents
    
    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
            
        X = cartesian position where magnetic field will be measured in GSM
            coordinates
            
    Outputs:
        df_b = dataframe with all the df data plus the delta B quantities
    """

    logging.info('Creating deltaB >rCurrents dataframe...')
    
    df_b = deepcopy(df)
    
    ##########################################################
    # Below we calculate the delta B in each cell from the
    # Biot-Savart Law.  We want the final result to be in nT.
    # dB = mu0/(4pi) (j x r)/r^3 dV
    #    = (4pi 10^(-7) [H/m])/(4pi) (10^(-6) [A/m^2]) [Re] [Re^3] / [Re^3]
    # where the fact that J is in microamps/m^2 and distances are in Re
    # is in the BATSRUS documentation.   We take Re = 6372 km = 6.371 10^6 m
    # dB = 6.371 10^(-7) [H/m][A/m^2][m]
    #    = 6.371 10^(-7) [H] [A/m^2]
    #    = 6.371 10^(-7) [kg m^2/(s^2 A^2)] [A/m^2]
    #    = 6.371 10^(-7) [kg / s^2 / A]
    #    = 6.371 10^(-7) [T]
    #    = 6.371 10^2 [nT]
    #    = 637.1 [nT] with distances in Re, j in microamps/m^2
    ##########################################################

    # Determine delta B in each cell
    df_b['r'] = ((X[0]-df_b['x'])**2+(X[1]-df_b['y'])**2+(X[2]-df_b['z'])**2)**(1/2)
    df_b['factor'] = 637.1*df_b['measure']/df_b['r']**3
    
    df_b['dBx'] = df_b['factor']*(df_b['jy']*(X[2]-df_b['z']) - df_b['jz']*(X[1]-df_b['y']))
    df_b['dBy'] = df_b['factor']*(df_b['jz']*(X[0]-df_b['x']) - df_b['jx']*(X[2]-df_b['z']))
    df_b['dBz'] = df_b['factor']*(df_b['jx']*(X[1]-df_b['y']) - df_b['jy']*(X[0]-df_b['x']))

    # Determine magnitude of dB
    df_b['dBmag'] = np.sqrt(df_b['dBx']**2 + df_b['dBy']**2 + df_b['dBz']**2)

    # Get the smallest cell (by volume), we will use it to normalize the
    # cells.  Cells far from earth are much larger than cells close to
    # earth.  That distorts some variables.  So we normalize the magnetic field
    # to the smallest cell.
    minMeasure = df_b['measure'].min()

    # Normalize magnetic field, as mentioned above
    df_b['dBmagNorm'] = df_b['dBmag'] * minMeasure/df_b['measure']
    df_b['dBxNorm'] = np.abs(df_b['dBx'] * minMeasure/df_b['measure'])
    df_b['dByNorm'] = np.abs(df_b['dBy'] * minMeasure/df_b['measure'])
    df_b['dBzNorm'] = np.abs(df_b['dBz'] * minMeasure/df_b['measure'])

    # Similarly, use dot-products to determine dBr, dBtheta, and dBphi
    df_b['dBr'] = df_b['dBx'] * np.sin(df_b['theta']) * np.cos(df_b['phi']) + \
        df_b['dBy'] * np.sin(df_b['theta']) * np.sin(df_b['phi']) + \
        df_b['dBz'] * np.cos(df_b['theta'])

    df_b['dBtheta'] = df_b['dBx'] * np.cos(df_b['theta']) * np.cos(df_b['phi']) + \
        df_b['dBy'] * np.cos(df_b['theta']) * np.sin(df_b['phi']) - \
        df_b['dBz'] * np.sin(df_b['theta'])

    df_b['dBphi'] = - df_b['dBx'] * \
        np.sin(df_b['phi']) + df_b['dBy'] * np.cos(df_b['phi'])

    # Determine delta B using the currents parallel and perpendicular to B. They
    # should sum to the delta B calculated above for the full current, jx, jy, jz
    df_b['dBparallelx'] = df_b['factor'] * \
        (df_b['jparallely']*(X[2]-df_b['z']) - df_b['jparallelz']*(X[1]-df_b['y']))
    df_b['dBparallely'] = df_b['factor'] * \
        (df_b['jparallelz']*(X[0]-df_b['x']) - df_b['jparallelx']*(X[2]-df_b['z']))
    df_b['dBparallelz'] = df_b['factor'] * \
        (df_b['jparallelx']*(X[1]-df_b['y']) - df_b['jparallely']*(X[0]-df_b['x']))

    df_b['dBperpendicularx'] = df_b['factor'] * \
        (df_b['jperpendiculary']*(X[2]-df_b['z']) - df_b['jperpendicularz']*(X[1]-df_b['y']))
    df_b['dBperpendiculary'] = df_b['factor'] * \
        (df_b['jperpendicularz']*(X[0]-df_b['x']) - df_b['jperpendicularx']*(X[2]-df_b['z']))
    df_b['dBperpendicularz'] = df_b['factor'] * \
        (df_b['jperpendicularx']*(X[1]-df_b['y']) - df_b['jperpendiculary']*(X[0]-df_b['x']))

    # Divide delta B from perpendicular currents into two - those along phi and 
    # everything else (residual)
    df_b['dBperpendicularphix'] =   df_b['factor']*df_b['jperpendicularphiy']*(X[2]-df_b['z'])
    df_b['dBperpendicularphiy'] = - df_b['factor']*df_b['jperpendicularphix']*(X[2]-df_b['z'])
    df_b['dBperpendicularphiz'] = df_b['factor']*(df_b['jperpendicularphix']*(X[1]-df_b['y']) - \
                                              df_b['jperpendicularphiy']*(X[0]-df_b['x']))
        
    df_b['dBperpendicularphiresx'] = df_b['factor']*(df_b['jperpendicularphiresy']*(X[2]-df_b['z']) - \
                                                  df_b['jperpendicularz']*(X[1]-df_b['y']))
    df_b['dBperpendicularphiresy'] = df_b['factor']*(df_b['jperpendicularz']*(X[0]-df_b['x']) - \
                                                  df_b['jperpendicularphiresx']*(X[2]-df_b['z']))
    df_b['dBperpendicularphiresz'] = df_b['factor']*(df_b['jperpendicularphiresx']*(X[1]-df_b['y']) - \
                                                  df_b['jperpendicularphiresy']*(X[0]-df_b['x']))

    return df_b

def create_cumulative_sum_dataframe(df):
    """Convert the dataframe with BATSRUS data to a dataframe that provides a 
    cumulative sum of dB in increasing r. To generate the cumulative sum, we 
    order the cells in terms of range r starting at the earth's center.  We 
    start the sum with a small sphere and vector sum all of the dB contributions 
    inside the sphere.  We then expand the sphere slightly and add to the sum.  
    Repeat until all cells are in the sum.

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities from
            convert_BATSRUS_to_dataframe
            
    Outputs:
        df_r = dataframe with all the df data plus the cumulative sums
    """

    logging.info('Creating cumulative sum dataframe...')

    # Sort the original data by range r, ascending.  Then calculate the total B 
    # based upon summing the vector dB values starting at r=0 and moving out.
    # Note, the dB for radii smaller than rCurrents should be 0, see
    # calculation of dBxyz above.

    df_r = deepcopy(df)
    df_r = df_r.sort_values(by='r0', ascending=True)
    df_r['dBxSum'] = df_r['dBx'].cumsum()
    df_r['dBySum'] = df_r['dBy'].cumsum()
    df_r['dBzSum'] = df_r['dBz'].cumsum()
    df_r['dBSumMag'] = np.sqrt(
        df_r['dBxSum']**2 + df_r['dBySum']**2 + df_r['dBzSum']**2)

    # Do cumulative sums on currents parallel and perpendicular to the B field
    df_r['dBparallelxSum'] = df_r['dBparallelx'].cumsum()
    df_r['dBparallelySum'] = df_r['dBparallely'].cumsum()
    df_r['dBparallelzSum'] = df_r['dBparallelz'].cumsum()
    df_r['dBparallelSumMag'] = np.sqrt(df_r['dBparallelxSum']**2
                                        + df_r['dBparallelySum']**2
                                        + df_r['dBparallelzSum']**2)

    df_r['dBperpendicularxSum'] = df_r['dBperpendicularx'].cumsum()
    df_r['dBperpendicularySum'] = df_r['dBperpendiculary'].cumsum()
    df_r['dBperpendicularzSum'] = df_r['dBperpendicularz'].cumsum()
    df_r['dBperpendicularSumMag'] = np.sqrt(df_r['dBperpendicularxSum']**2
                                            + df_r['dBperpendicularySum']**2
                                            + df_r['dBperpendicularzSum']**2)

    df_r['dBperpendicularphixSum'] = df_r['dBperpendicularphix'].cumsum()
    df_r['dBperpendicularphiySum'] = df_r['dBperpendicularphiy'].cumsum()
    df_r['dBperpendicularphizSum'] = df_r['dBperpendicularphiz'].cumsum()

    df_r['dBperpendicularphiresxSum'] = df_r['dBperpendicularphiresx'].cumsum()
    df_r['dBperpendicularphiresySum'] = df_r['dBperpendicularphiresy'].cumsum()
    df_r['dBperpendicularphireszSum'] = df_r['dBperpendicularphiresz'].cumsum()

    return df_r

def create_jrtp_cdf_dataframes(df):
    """Use the dataframe with BATSRUS dataframe to develop jr, jtheta, jphi
    cummulative distribution functions (CDFs).

    Inputs:
        df = dataframe with BATSRUS and other calculated quantities
        
    Outputs:
        cdf_jr, cdf_jtheta, cdf_jphi = jr, jtheta, jphi CDFs.
    """

    logging.info('Creating CDF dataframe...')

    # Sort the j components (jr, jtheta, jphi) in order of value.  If each entry 
    # in the dataframe is equally likely, the arange is the CDF.

    df_jr = deepcopy(df)
    df_jtheta = deepcopy(df)
    df_jphi = deepcopy(df)
    
    df_jr = df_jr.sort_values(by='jr', ascending=True)
    df_jr['cdf'] = np.arange(1, len(df_jr)+1)/float(len(df_jr))

    df_jtheta = df_jtheta.sort_values(by='jtheta', ascending=True)
    df_jtheta['cdf'] = np.arange(1, len(df_jtheta)+1)/float(len(df_jtheta))

    df_jphi = df_jphi.sort_values(by='jphi', ascending=True)
    df_jphi['cdf'] = np.arange(1, len(df_jphi)+1)/float(len(df_jphi))

    return df_jr, df_jtheta, df_jphi


def calc_gap_dB(XGSM, base, dirpath, rCurrents, rIonosphere, nTheta, nPhi, nTheta0):
    """Process data in BATSRUS file to calculate the delta B at point XGSM as
    determined by the field-aligned currents between the radius rCurrents
    and the rIonosphere.  Biot-Savart Law is used for calculation.  We will integrate
    across all currents flowing through the sphere at range rCurrents from earth
    origin.
    
    Inputs:
        XGSM = GSM position where magnetic field will be measured, BATSRUS data
            is in GSM coordinates.  Dipole data is in SM coordinates
        
        base = basename of BATSRUS file.  Complete path to file is:
            dirpath + base + '.out'
            
        dirpath = path to directory containing base
        
        rCurrents = range from earth center below which results are not valid
            measured in Re units
            
        rIonosphere = equal range from earth center to the ionosphere, measured
            in Re units (1.01725 in magnetopost code)
            
        nTheta, nPhi, nTheta0 = number of points to be examined in the 
            numerical integration, e.g., nPhi points along phi.
            
    Outputs:
        dBGSM = deltaB due to field-aligned currents (in GSM coordinates)
            
        title = title to use in plots, which is derived from base (file basename)
    """

    logging.info(f'Calculate FAC dB...{base}')

    # Verify BATSRUS file exists
    assert exists(dirpath + base + '.out')
    assert exists(dirpath + base + '.info')
    assert exists(dirpath + base + '.tree')

    # Read BATSRUS file
    swmfio.logger.setLevel(logging.INFO)
    batsrus = swmfio.read_batsrus(dirpath + base)
    assert(batsrus != None)

    # Set up some variables used below
    bGSM = np.empty(3)
    jGSM = np.empty(3)
    dBSM = np.zeros(3)
    xSM = np.empty(3)
    x_facSM = np.zeros(3)
    b_facSM = np.zeros(3)

    ############################################################
    # Earth dipole field is in SM coordinates, and BATSRUS is in GSM coordinates
    # We use SM or GSM in the variable names to identify the applicable
    # coordinate system.  A few items do not have SM or GSM and are scalars
    # independent of coordinate system
    ############################################################

    # Get a matrix to transform from SM to GSM, the transpose is the inverse operation.
    y, m, d, hh, mm, ss = util.date_time(base)
    sm2gsm = coord.get_transform_matrix([y,m,d,hh,mm,ss], 'SM', 'GSM')
    
    # Transform the point where the field is measured to SM coordinates, we
    # will need it below.  Its done here to get it out of the integration loop.
    XSM = np.dot( sm2gsm.T, XGSM)

    # Determine the size of the steps in thetaSM and phiSM for numerical integration
    # over sphere at rCurrents
    dThetaSM = np.pi    / nTheta
    dPhiSM   = 2.*np.pi / nPhi
    
    # Start the loops for the Biot-Savart numerical integration.  We integrate
    # over a spherical shell between rIonosphere and rCurrents.  We use three 
    # loops - theta, phi and theta0.  theta and phi cover the outer boundary
    # where integration begins (the sphere at rCurrents).  theta0 integrates over
    # each field line as the current follows field lines down to rIonosphere.  
    # We do the integration in the earth dipole field coordinate system (SM).
    for i in range(nTheta):
        # thetaSM is a latitude from pi/2 -> -pi/2
        # Find thetaSM at the middle of each dSurfaceSM element
        # dSurfaceSM is from thetaSM - dThetaSM/2 to thetaSM + dThetaSM/2
        thetaSM = np.pi/2 - (i + 0.5) * dThetaSM
        
        # Surface area of element on sphere at radius rCurrents
        # thetaSM is latitude, so use cos(thetaSM)
        dSurfaceSM = rCurrents**2 * np.cos(thetaSM) * dThetaSM * dPhiSM

        for j in range(nPhi):
            # Find phiSM at the middle of each dSurfaceSM element
            # dSurfaceSM is from phi - dPhi/2 to phi + dPhi/2
            phiSM = (j + 0.5) * dPhiSM

            # Find the cartesian coordinates for point at rCurrents, thetaSM, phiSM
            # This is the point where we start the integration along a field line.
            # It is in SM coordinates, we will need to convert to GSM coordinates
            # to get BATSRUS data.  Remember thetaSM is latitude, so sin and cos
            # are switched
            xSM[0] = rCurrents * np.cos(phiSM) * np.cos(thetaSM)
            xSM[1] = rCurrents * np.sin(phiSM) * np.cos(thetaSM)
            xSM[2] = rCurrents * np.sin(thetaSM)
            xGSM = np.dot( sm2gsm, xSM )

            # Use BATSRUS interpolator to get b and j at point xGSM
            bGSM[0] = batsrus.interpolate(xGSM, 'bx')
            bGSM[1] = batsrus.interpolate(xGSM, 'by')
            bGSM[2] = batsrus.interpolate(xGSM, 'bz')
            jGSM[0] = batsrus.interpolate(xGSM, 'jx')
            jGSM[1] = batsrus.interpolate(xGSM, 'jy')
            jGSM[2] = batsrus.interpolate(xGSM, 'jz')
                        
            ################################################################
            # Below we find the Field Aligned Current (FAC).  We can safely
            # ignore the current perpendicular to the magnetic field.  Lotko
            # shows j_perp = 0 and j_parallel/B_0 = constant in the ionosphere
            # and low-altitude magnetosphere
            #
            # See Lotko, 2004, J. Atmo. Solar-Terrestrial Phys., 66, 1443–1456
            ################################################################
            
            # We want the Field Aligned Current (FAC). Because the B field can
            # be perpendicular to the sphere at rCurrents (at north-south poles)
            # or tangent to it at the equator, we take the current density
            # at rCurrents and use the dot product to find the portion parallel 
            # to B field (b-hat).  We use the GSM coordinates of the BATSRUS data.
            # An analysis of the fraction of the current density parallel
            # to the B field is 75 to 95% averaged over the rCurrents sphere.
            # This analysis looked at some of the colaba SWMF runs.  See
            # code in compare_jFAC_initialization.py
            b_hatGSM = bGSM / np.linalg.norm(bGSM)
            
            #!!!!!!!!!
            r_hatGSM = xGSM / np.linalg.norm(xGSM)
            
            # Note, j_fac_mag, the magnitude of the FAC current density,
            # is independent of coordinate system, its a scalar
            j_fac_mag = np.dot(jGSM, b_hatGSM)
            
            #!!!!!!!!!!!
            # j_fac_mag = j_fac_mag * np.dot(b_hatGSM, r_hatGSM)
            
            ##########################################################
            # To do the numerical integration, we walk down each field line 
            # that starts at rCurrents, thetaSM, and phiSM.  We will need 
            # points along a dipole field line to map the current toward the 
            # earth's surface.  To do so, we consider r (distance from earth
            # center to field line) as a function of theta0 (latitude to point
            # on the field line measured in SM cooridnates) 
            #
            # See Willis and Young, 1987, Geophys. J.R. Astr. Soc. 89, 1011-1022
            # Equation for the field lines of an axisymmetric magnetic dipole
            #
            #          r = r1*cos(theta0)**2  
            #
            # where r1 is radius to field line at equator. Note, sin became cos 
            # because we use latitude.  We will step down the field line in 
            # nTheta0 steps.  And we will use the above equation to determine 
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

            # Max and min of theta0SM are at rCurrents and rIonosphere.  We
            # need to make sure that we have the correct sign for theta0minSM.
            # It has the same sign as theta0maxSM.  We will divide this range 
            # into segments for Biot-Savart integration
            
            theta0maxSM = thetaSM
            r1SM = rCurrents / np.cos(theta0maxSM)**2
            theta0minSM = np.sign( theta0maxSM ) * np.arccos( np.sqrt(rIonosphere/r1SM) )
            dTheta0SM = (theta0maxSM - theta0minSM) / nTheta0
            
            # Do integration along field line.
            for k in range(nTheta0):
                # Find theta0 at the middle of each field line segment
                theta0SM = theta0maxSM - (k + 0.5) * dTheta0SM
                
                # Arc length of field line segment, see Chapman and Sugiura above.
                dsSM = r1SM * np.sqrt(1 + 3*np.sin(theta0SM)**2) * np.cos(theta0SM) * dTheta0SM

                # FAC follows B dipole field, see Willis and Young above.
                # Use this equation to determine x,y,z of new point on field line
                rSM = r1SM * np.cos(theta0SM)**2
                x_facSM[0] = rSM * np.cos(phiSM) * np.cos(theta0SM)
                x_facSM[1] = rSM * np.sin(phiSM) * np.cos(theta0SM)
                x_facSM[2] = rSM * np.sin(theta0SM)
                
                # Calculate earth's magnetic field in cartesian coordinates using a 
                # simple dipole model
                #
                # https://en.wikipedia.org/wiki/Dipole_model_of_the_Earth%27s_magnetic_field
                #
                # As noted in Lotko, magnetic perturbations in the ionosphere 
                # and the low-altitude magnetosphere are much smaller than 
                # the geomagnetic field B0.  So we can use the simple dipole field.
                B0 = 3.12e+4 # Changed to nT units             
                b_facSM[0] = - 3 * B0 * x_facSM[0] * x_facSM[2] / rSM**5
                b_facSM[1] = - 3 * B0 * x_facSM[1] * x_facSM[2] / rSM**5
                b_facSM[2] = - B0 * ( 3 * x_facSM[2]**2 - rSM**2 ) / rSM**5

                # Get the unit vector for b_facSM, we use it below
                # to determine the direction of the FAC current flow.
                b_fac_hatSM = b_facSM / np.linalg.norm(b_facSM)
                
                # Determine range to XSM where magnetic field is measured
                # Note, XSM is XGSM in SM coordinates
                r_facSM = XSM - x_facSM
                r_fac_magSM = np.linalg.norm( r_facSM )

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
                
                ##########################################################
                # In Biot-Savart, we use that j_fac_mag * dSurface is constant.
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
                
                dBSM[:] = dBSM[:] + 637.1 * j_fac_mag * np.cross( b_fac_hatSM, r_facSM ) \
                    * dSurfaceSM * dsSM / r_fac_magSM**3

                
    # dB calculation above is in SM coordinates, we want GSM for use with BATSRUS data
    dBGSM = np.dot( sm2gsm.T, dBSM )
    
    # Create the title that we'll use in the graphics
    words = base.split('-')
    title = 'Time: ' + words[1] + ' (hhmmss)'

    return dBGSM, title    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 16:46:59 2023

@author: Dean Thomas
"""

##########################################################################
##########################################################################
# Compare two different ways of initalizing Field Aligned Currents at 
# r = rCurrents.  My initial assumption was to find jr at rCurrents 
# (from j dot r-hat), then find the portion of jr parallel to the B field at 
# rCurrents.  However, this may not make sense.  The B field can range from 
# radially aligned to tanget to the sphere at rCurrents.  So we also consider 
# j dot B-hat.
##########################################################################
##########################################################################

import logging
import swmfio
from copy import deepcopy
from os.path import exists
import pandas as pd
import numpy as np
from hxform import hxform as hx

COLABA = True

# origin and target define where input data and output plots are stored
if COLABA:
    ORIGIN = '/Volumes/Physics HD v2/runs/DIPTSUR2/GM/IO2/'
    TARGET = '/Volumes/Physics HD v2/runs/DIPTSUR2/plots/'
else:
    ORIGIN = '/Volumes/Physics HD v2/divB_simple1/GM/'
    TARGET = '/Volumes/Physics HD v2/divB_simple1/plots/'

def date_time(file):
    """Pull date and time from file basename

    Inputs:
        file = basename of file.
        
    Outputs:
        month, day, year, hour, minute, second 
     """
    words = file.split('-')

    date = int(words[0].split('e')[1])
    y = date//10000
    n = (date % 10000) // 100
    d = date % 100

    # Convert time to a integers
    t = int(words[1])
    h = t//10000
    m = (t % 10000) // 100
    s = t % 100

    # logging.info(f'Time: {date} {t} Year: {y} Month: {n} Day: {d} Hours: {h} Minutes: {m} Seconds: {s}')

    return y, n, d, h, m, s

def get_files(orgdir=ORIGIN, base='3d__*'):
    """Create a list of files that we will process.  Look in the basedir directory,
    and get list of file basenames.

    Inputs:
        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out'
            
        orgdir = path to directory containing input files
        
    Outputs:
        l = list of file basenames.
    """
    import os
    import glob

    # Create a list of files that we will process
    # Look in the basedir directory.  Get list of file basenames

    # In this version, we find all of the base + '.out' files
    # and retrieve their basenames
    os.chdir(orgdir)

    l1 = glob.glob(base + '.out')

    # Strip off extension
    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    # Colaba incliudes 697 files, reduce the number by
    # accepting those only every 15 minutes
    if COLABA: 
        l2 = deepcopy(l1) 
        for i in range(len(l2)):
            y,n,d,h,m,s = date_time(l2[i])
            if( m % 15 != 0 ):
                l1.remove(l2[i])

    l1.sort()

    return l1

def compare_inital_FAC(XGSM, base, dirpath, rCurrents, rIonosphere, nTheta, nPhi, nTheta0):
    """Process data in BATSRUS file to compare two different ways of initalizing
    Field Aligned Currents at r = rCurrents.  My initial assumption was to find
    jr at rCurrents (from j dot r-hat), then find the portion of jr parallel to
    the B field at rCurrents.  However, this may not make sense.  The B field can
    range from radially aligned to tanget to the sphere at rCurrents.  So we also 
    consider j dot B-hat.
    
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
    xSM = np.empty(3)
    jdotbfrac = np.zeros(nTheta*nPhi)
    j_facfrac = np.zeros(nTheta*nPhi)

    ############################################################
    # Earth dipole field is in SM coordinates, and BATSRUS is in GSM coordinates
    # We use SM or GSM in the variable names to identify the applicable
    # coordinate system.  A few items do not have SM or GSM and are scalars
    # independent of coordinate system
    ############################################################

    # Get a matrix to transform from SM to GSM, the transpose is the inverse operation.
    
    ############################################################
    ############################################################
    #!!! Hack - time is fixed, it should be dependent on the file
    ############################################################
    ############################################################
    time = (2011,1,1)
    sm2gsm = hx.get_transform_matrix(time, 'SM', 'GSM')
    
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
        # dSurfaceSM = rCurrents**2 * np.cos(thetaSM) * dThetaSM * dPhiSM

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
            
            # We want the Field Aligned Current (FAC) penetrating into the 
            # sphere at radius rCurrents.  We use a dot product to find the
            # portion of j parallel to r-hat.  Then use a second dot product 
            # to find the portion parallel to B field (b-hat).  We use the GSM
            # coordinates of the BATSRUS data            
            r_hatGSM = xGSM / rCurrents
            b_hatGSM = bGSM / np.linalg.norm(bGSM)
            
            # Note, j_fac_mag, the magnitude of the FAC current density,
            # is independent of coordinate system, its a scalar
            j_fac_mag = np.dot(jGSM, r_hatGSM) * np.dot(b_hatGSM, r_hatGSM)
            
            jGSMmag = np.linalg.norm(jGSM)
            jdotbfrac[i*nTheta+j]= abs(np.dot(jGSM, b_hatGSM)/jGSMmag)
            j_facfrac[i*nTheta+j] = abs(j_fac_mag/jGSMmag)
            # print( '++++++++++ {:2.2}'.format(jdotbfrac[i*nTheta+j]))

    # print('====================== Avg and StdDev of jdotbfrac {:2.2} {:2.2}'.format( np.mean(jdotbfrac), np.std(jdotbfrac) ) )
    return np.mean(jdotbfrac), np.std(jdotbfrac), np.mean(j_facfrac), np.std(j_facfrac)

def main():
    """Compare two different ways of initalizing Field Aligned Currents at 
    r = rCurrents.  My initial assumption was to find jr at rCurrents 
    (from j dot r-hat), then find the portion of jr parallel to the B field at 
    rCurrents.  However, this may not make sense.  The B field can range from 
    radially aligned to tanget to the sphere at rCurrents.  So we also consider 
    j dot B-hat.
    
    The output is two plots showing the fraction of j that will go into jFAC initial.
    How that fraction varies over time will be plotted.
    """
    
    nTheta = 30
    nPhi = 30
    nTheta0 = 30
    rCurrents = 2.
    rIonosphere = 1.01725

    if COLABA:
        files = get_files(base='3d__var_2_e*')
    else:
        files = get_files(base='3d__*')
        
    logging.info('Num. of files: ' + str(len(files)))

    X = [1, 0, 0]
    l = len(files)
    # l = 5
    jdotbfracmean = np.zeros(l)
    jdotbfracstd  = np.zeros(l)
    j_facfracmean = np.zeros(l)
    j_facfracstd  = np.zeros(l) 
    times         = np.zeros(l)

    for i in range(l):
        jdotbfracmean[i], jdotbfracstd[i], j_facfracmean[i], j_facfracstd[i] = \
            compare_inital_FAC(X, files[i], ORIGIN, rCurrents, rIonosphere, nTheta, nPhi, nTheta0)
        y, m, d, hh, mm, ss = date_time(files[i])
        times[i] = hh + mm/60.
        
    df = pd.DataFrame({'times':times, 'j dot b mean':jdotbfracmean, 'j dot b std':jdotbfracstd, \
                       'j_fac mean':j_facfracmean, 'j_fac std':j_facfracstd})
    # df.plot( 'times', 'j dot b mean' )
    # df.plot( 'times', 'j dot b std' )
    # df.plot( 'times', 'j_fac mean' )
    # df.plot( 'times', 'j_fac std' )
    df.plot( 'times', 'j dot b mean', yerr= 'j dot b std', ylim=(0,1),
            xlabel='Time (hr)', ylabel='Fraction of |j|')
    df.plot( 'times', 'j_fac mean', yerr = 'j_fac std', ylim=(0,1),
            xlabel='Time (hr)', ylabel='Fraction of |j|')
    
if __name__ == "__main__":
   main()


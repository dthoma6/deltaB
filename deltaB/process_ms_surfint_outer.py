#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:14:44 2024

@author: Dean Thomas
"""

# from numba import jit
import swmfio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy.time import Ticktock
import os.path

from deltaB.util import create_directory, date_timeISO
from deltaB.coordinates import GSMtoSM, iso2ints, get_NED_components
from deltaB.BATSRUS_interpolator import BATSRUS_interpolator

# Set to True to use Kamodo linear interpolation (preferred)
# Set False for swmfio interpolation
KAMODO=True

# @jit(nopython=True)
def calc_ms_surfint_outer_b_sub(XGSM, timeISO, batsrus, nX=100, nY=100, nZ=100):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a BATSRUS file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a surface integral on outer boundary of BATSRUS grid.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data from swmfio
        
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
    
    # Create Kamodo BATSRUS interpolators, see BATSRUS_interpolator.py
    if KAMODO:
        batsrus_interp = BATSRUS_interpolator(batsrus)
        batsrus_interp.register_variable( 'bx' )
        batsrus_interp.register_variable( 'by' )
        batsrus_interp.register_variable( 'bz' )
    
    # Local routine that is used in loops below to calculate contribution
    # from each surface element.
    def calc( xx, xxhat, dS ):
        """ xx = point in space (GSM)
            xxhat = unit vector for surface
            dS = size of surface element
        """
        # Get B field at point x (in GSM coordinates)
        if KAMODO:
            # Kamodo linear interpolation (Preferred)
            Bpt[0] = batsrus_interp.interp(xx, 'bx')[0]
            Bpt[1] = batsrus_interp.interp(xx, 'by')[0]
            Bpt[2] = batsrus_interp.interp(xx, 'bz')[0]
        else:
            # swmfio interpolation, which is simplistic
            Bpt[0] = batsrus.interpolate(xx, 'bx')
            Bpt[1] = batsrus.interpolate(xx, 'by')
            Bpt[2] = batsrus.interpolate(xx, 'bz')
            
        if( np.isnan(Bpt[0]) or np.isnan(Bpt[1]) or np.isnan(Bpt[2]) ):
            print(xx, xxhat, Bpt)
            assert(False)
                    
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
            calc( x, xhat, dSxy )

            # bottom face
            x = np.array([xloop, yloop, minZ])
            xhat = np.array([0.,0.,-1.])
            calc( x, xhat, dSxy )
            
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
            calc( x, xhat, dSxz )

            # right face
            x = np.array([xloop, minY, zloop])
            xhat = np.array([0.,-1.,0.])
            calc( x, xhat, dSxz )

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
            calc( x, xhat, dSyz )
            
            # back face
            x = np.array([minX, yloop, zloop])
            xhat = np.array([-1.,0.,0.])
            calc( x, xhat, dSyz )
            
    # Add irrotational and solenoidal contributions to get total B contribution
    B[:] = Birr[:] + Bsol[:]
    
    return B, Birr, Bsol
  
def calc_ms_surfint_outer_b(XGSM, timeISO, batsrus, nX=100, nY=100, nZ=100):
    """Process data in BATSRUS file to calculate the delta B at point XGSM.
    Helmholtz decomposition theorem used to convert Biot-Savart Law to a 
    surface integral used for calculation.  We will integrate across the outer
    boundary of the BATSRUS grid.  
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        batsrus = BATSRUS data from swmfio
        
        timeISO = ISO time for data in BATSRUS file
              
        nX, nY, nZ = number of steps in numerical integration over outer faces,
            e.g., nX*nY points on outer surfaces parallel to X-Y plane

    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(r'Calculate magnetosphere outer surface integral dB...')

    # We need the time to switch from GSM to SM coordinates
    time = iso2ints( timeISO )

    # Results in GSM coordinates
    BGSM = np.zeros(3)
    BirrGSM = np.zeros(3)
    BsolGSM = np.zeros(3)

    # Do surface integral
    BGSM, BirrGSM, BsolGSM = calc_ms_surfint_outer_b_sub(XGSM, timeISO, batsrus, 
                                                   nX, nY, nZ)
    # Convert to SM coordinates        
    B = np.zeros(3)
    Birr = np.zeros(3)
    Bsol = np.zeros(3)

    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')

    B = GSMtoSM( BGSM, time, ctype_in='car', ctype_out='car')
    Birr = GSMtoSM( BirrGSM, time, ctype_in='car', ctype_out='car')
    Bsol = GSMtoSM( BsolGSM, time, ctype_in='car', ctype_out='car')
    
    Bn, Be, Bd = get_NED_components( B, XSM )
    Birrn, Birre, Birrd = get_NED_components( Birr, XSM )
    Bsoln, Bsole, Bsold = get_NED_components( Bsol, XSM )
    
    return Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, B[0], B[1], B[2]      


# Example info.  Info is used below in call to loop_ms_b
# info = {
#         "model": "SWMF",
#         "run_name": "SWPC_SWMF_052811_2",
#         "rCurrents": 4.0,
#         "file_type": "cdf",
#         "dir_run": os.path.join(data_dir, "SWPC_SWMF_052811_2"),
#         "dir_plots": os.path.join(data_dir, "SWPC_SWMF_052811_2.plots"),
#         "dir_derived": os.path.join(data_dir, "SWPC_SWMF_052811_2.derived"),
#         "deltaB_files": {
#             "YKC": os.path.join(data_dir, "SWPC_SWMF_052811_2", "2006_YKC_pointdata.txt")
#         }
# }

def loop_ms_surfint_outer_b(info, point, reduce, nX=100, nY=100, nZ=100, 
                      deltahr=None, maxcores=20, deltaBlist=False):
    """Use surface integral at outer boundary from Helmholtz Decomposition Theorem 
    in calc_ms_surfint_outer_b to determine the magnetic field (in 
    North-East-Down coordinates) at magnetometer point.  Surface integral uses 
    magnetosphere current density as defined in BATSRUS files

    Inputs:
        info = information on BATSRUS data, see example immediately above
        
        point = string identifying magnetometer location.  The actual location
            is pulled from a list
            
        reduce = Do we skip files to save time.  If None, do all files.  If not
            None, then its a integer that determine how many files are skipped
        
        nX, nY, nZ = number of steps in numerical integration over outer faces,
            e.g., nX*nY points on outer surfaces parallel to X-Y plane

        deltahr = if None ignore, if number, shift ISO time by that 
            many hours.  If value given, must be float.
            
        maxcores = for parallel processing, the maximum number of cores to use
        
        deltaBlist = Boolean.  False use magnetpost list of magnetometer sites.
            True use deltaB list of magnetometer sites.
        
    Outputs:
        time, Bn, Be, Bd = saved in pickle file
    """
    # Wrapper function that contains the bulk of the routine, used
    # for parallel processing of the data
    def wrap_ms( i, times, deltahr, XGEO, info ):
        time = times[i]
        
        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[i]]
        base = os.path.basename(filepath)

        logging.info(f'Calculate magnetosphere outer surface integral dB for... {base}')
        
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
        
        # Get the magnetometer position, X, in GSM coordinates for compatibility with
        # BATSRUS data
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        XGSM = XGEO.convert( 'GSM', 'car' )
        X = XGSM.data[0]
    
        # Read in the BATSRUS file 
        batsrus = swmfio.read_batsrus(filepath)
    
        # Use Helmholtz decomposition surface integral to calculate magnetic 
        # field, B, at magnetometer position X (GSM).  Store the results, which 
        # are in SM coordinates, and the time
        Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
                Bx, By, Bz = calc_ms_surfint_outer_b(X, timeISO, batsrus, 
                                               nX, nY, nZ)
        
        return Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
                Bx, By, Bz, Btime

    # Verify input parameters
    assert isinstance(point, str)

    # Make sure delta_hr is float
    if deltahr is not None:
        assert( type(deltahr) == float )
 
    # Get times for BATSRUS files, if reduce is True we reduce the number of 
    # files selected.  info parameters define location (dir_run) and file types    
    times = list(info['files']['magnetosphere'].keys())
    if reduce != None:
        assert isinstance( reduce, int )
        times = times[0:len(times):reduce]
    n = len(times)

    # We need the magnetometer coordinates at point.  Either look it up
    # in the magnetopost list or in deltaB list
    from spacepy import coordinates as coord
    if deltaBlist == False:
        # Get the magnetometer location using magnetopost list
        from magnetopost.config import defined_magnetometers
        pointX = defined_magnetometers[point]
        XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)
    else:
        # Get the magnetometer location from the deltaB list
        from deltaB.magnetometers import specified_magnetometers
        pointX = specified_magnetometers[point]
        XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)
        
    # Loop through the files using parallel processing
    if maxcores > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        num_cores = min(num_cores, len(times), maxcores)
        logging.info(f'Parallel processing {len(times)} timesteps using {num_cores} cores')
        results = Parallel(n_jobs=num_cores)(delayed(wrap_ms)( p, times, deltahr, XGEO, info ) 
                                   for p in range(len(times)))
        
        Bn, Be, Bd, Birrn, Birre, Birrd, Bsoln, Bsole, Bsold, \
                Bx, By, Bz, Btimes = zip(*results)
        
    # Loop through files if no parallel processing
    else:
        # Prepare storage of variables
        Bn = np.zeros(n)
        Be = np.zeros(n)
        Bd = np.zeros(n)
        Birrn = np.zeros(n)
        Birre = np.zeros(n)
        Birrd = np.zeros(n)
        Bsoln = np.zeros(n)
        Bsole = np.zeros(n)
        Bsold = np.zeros(n)
        Bx = np.zeros(n)
        By = np.zeros(n)
        Bz = np.zeros(n)
        
        Btimes = [None] * n

        for p in range(len(times)):
            Bn[p], Be[p], Bd[p], \
                Birrn[p], Birre[p], Birrd[p], \
                Bsoln[p], Bsole[p], Bsold[p], \
                Bx[p], By[p], Bz[p], Btimes[p] = \
                wrap_ms( p, times, deltahr, XGEO, info ) 

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
                'Birrn': Birrn, 'Birre': Birre, 'Birrd': Birrd, 
                'Bsoln': Bsoln, 'Bsole': Bsole, 'Bsold': Bsold, 
                'Bx': Bx, 'By': By, 'Bz': Bz, 
                r'Time (hr)': Btimes, r'Datetime': dtimes,
                r'Month': dtimes_m, r'Day': dtimes_d,
                r'Hour': dtimes_hh, r'Minute': dtimes_mm}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    if KAMODO:
        pklname = 'dB_si_msph_outer-' + point + '.pkl'
    else:
        pklname = 'dB_si_msph_swmfio_outer-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    

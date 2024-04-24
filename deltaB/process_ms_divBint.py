#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:14:44 2024

@author: Dean Thomas
"""

from numba import jit
import swmfio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from spacepy.time import Ticktock
import os.path

from deltaB.util import create_directory, get_NED_components, date_timeISO
from deltaB.coordinates import GSMtoSM, iso2ints

# If SECOND_ORDER is True, use 2nd order stencils for derivatives, otherwise
# use swmfio get_native_partial_derivatives
SECOND_ORDER=True

# If USE_B1 is True, use b1x, b1y, and b1z in BATSRUS data.  Otherwise use
# bx, by, bz in BATSRUS
USE_B1=True

@jit(nopython=True)
def calcDivB(batsrus, i, j, k, n, nI, nJ, nK, dX, dY, dZ, _bx, _by, _bz):
    """ Subroutine for calc_ms_divBint_b_sub that allows numba accelleration.  It  
    calculates divergence of B at point i,j,k in block n using data from 
    a BATSRUS file and the Helmholtz decompostion theorem to replace Biot-Savart 
    with a volume integral over the divergence of B.
    
    Inputs:
        batsrus = BATSRUS data from swmfio
        
        i,j,k = grid coordinates of point inside block n
        
        nI,nJ,nK = number of x,y,z points, respectively, in block n. Provided to 
            avoid constantly looking them up
        
        dX,dY,dZ = distance between two consecutive points along x,y,z axis,
            respectively, for points inside block n.   Provided to avoid 
            constantly calculating them.
        
        _bx,_by,_bz = batsrus.varidx values for bx, by, bz.  Provided to avoid 
            constantly looking them up
                       
    Outputs:
        divB = divergence of B at point i,j,k in block n, which is sum of 
            divBx, divBy and divBz (in GSM coordinates)
    """
    
    assert( i>=0 and i<nI )
    assert( j>=0 and j<nJ )
    assert( k>=0 and k<nK )
    
    if SECOND_ORDER:
        # Use 2nd order stencils to calculate derivatives, sum derivatives to 
        # determine divB

        if i > 0 and i < nI-1: # in interior of block n
            divBx = (batsrus.DataArray[_bx, i+1, j, k, n] - batsrus.DataArray[_bx, i-1, j, k, n])/(2*dX)
        elif i == 0: # on face
            divBx = (-3*batsrus.DataArray[_bx, 0, j, k, n] + 4*batsrus.DataArray[_bx, 1, j, k, n]
                    - batsrus.DataArray[_bx, 2, j, k, n])/(2*dX)
        else: # i == nI-1: on face
            divBx = (3*batsrus.DataArray[_bx, nI-1, j, k, n] - 4*batsrus.DataArray[_bx, nI-2, j, k, n]
                    + batsrus.DataArray[_bx, nI-3, j, k, n])/(2*dX)        
        
        if j > 0 and j < nJ-1: # in interior of block n
            divBy = (batsrus.DataArray[_by, i, j+1, k, n] - batsrus.DataArray[_by, i, j-1, k, n])/(2*dY)
        elif j == 0: # on face
            divBy = (-3*batsrus.DataArray[_by, i, 0, k, n] + 4*batsrus.DataArray[_by, i, 1, k, n]
                    - batsrus.DataArray[_by, i, 2, k, n])/(2*dY)
        else: # j == nJ-1: on face
            divBy = (3*batsrus.DataArray[_by, i, nJ-1, k, n] - 4*batsrus.DataArray[_by, i, nJ-2, k, n]
                    + batsrus.DataArray[_by, i, nJ-3, k, n])/(2*dY)        
        
        if k > 0 and k < nK-1: # in interior of block n
            divBz = (batsrus.DataArray[_bz, i, j, k+1, n] - batsrus.DataArray[_bz, i, j, k-1, n])/(2*dZ)
        elif k == 0: # on face
            divBz = (-3*batsrus.DataArray[_bz, i, j, 0, n] + 4*batsrus.DataArray[_bz, i, j, 1, n]
                    - batsrus.DataArray[_bz, i, j, 2, n])/(2*dZ)
        else: # k == ni-1: on face
            divBz = (3*batsrus.DataArray[_bz, i, j, nK-1, n] - 4*batsrus.DataArray[_bz, i, j, nK-2, n]
                    + batsrus.DataArray[_bz, i, j, nK-3, n])/(2*dZ)        
        
        divB = divBx + divBy + divBz
    else:
        # Use swmfio get_native_partial_derivatives to determine divB
        
        ind = i + nI*j + nI*nJ*k + nI*nJ*nK*n
        if USE_B1:
            partials_b1x = batsrus.get_native_partial_derivatives(ind, 'b1x')
            partials_b1y = batsrus.get_native_partial_derivatives(ind, 'b1y')
            partials_b1z = batsrus.get_native_partial_derivatives(ind, 'b1z')
        else:
            partials_b1x = batsrus.get_native_partial_derivatives(ind, 'bx')
            partials_b1y = batsrus.get_native_partial_derivatives(ind, 'by')
            partials_b1z = batsrus.get_native_partial_derivatives(ind, 'bz')
    
        divB = partials_b1x[0] + partials_b1y[1] + partials_b1z[2]
        
    return divB

@jit(nopython=True)
def calc_ms_divBint_b_sub(XGSM, timeISO, batsrus, rCurrents):
    """ Subroutine for calc_ms_surfint_b that allows numba accelleration.  It  
    calculates total B field at point XGSM using data from a BATSRUS file and the
    Helmholtz decompostion theorem to replace Biot-Savart volume integral with  
    a volume integral over divergence of B.
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data from swmfio
        
        rCurrents = range from earth center below which results are not valid.
            Measured in Re units.  We drop the data inside radius rCurrents
        
    Outputs:
        B = total B due to magnetospheric currents (in GSM coordinates)
    """

    # Set up some variables used below
    B = np.zeros(3)
    r = np.zeros(3)
    
    # We use these values throughout the routine, so to avoid muliple lookups
    # we look for them once.
    _x = batsrus.varidx['x']
    _y = batsrus.varidx['y']
    _z = batsrus.varidx['z']
    
    if USE_B1:
        _bx = batsrus.varidx['b1x']
        _by = batsrus.varidx['b1y']
        _bz = batsrus.varidx['b1z']
    else:
        _bx = batsrus.varidx['bx']
        _by = batsrus.varidx['by']
        _bz = batsrus.varidx['bz']
    
    _measure = batsrus.varidx['measure']
    
    nVar, nI, nJ, nK, nBlock = batsrus.DataArray.shape

    # Loop through each block, then loop through each point in the block.
    for n in range(nBlock):
        
        # Determine dX, dY, and dZ for this block
        dX = batsrus.DataArray[_x,1,0,0,n] - batsrus.DataArray[_x,0,0,0,n]
        dY = batsrus.DataArray[_y,0,1,0,n] - batsrus.DataArray[_y,0,0,0,n]
        dZ = batsrus.DataArray[_z,0,0,1,n] - batsrus.DataArray[_z,0,0,0,n]
        
        # Iterate thru points in block, calculating the divergence of B at
        # each point.  Use this in the divB integral from the Helmholtz
        # Decomposition Theorem to determine dB
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):
                    # Distance from center of earth to point i,j,k,n
                    r0 = np.sqrt(batsrus.DataArray[_x,i,j,k,n]**2 +
                                 batsrus.DataArray[_y,i,j,k,n]**2 +
                                 batsrus.DataArray[_z,i,j,k,n]**2)
                    
                    # Only include point if it is outside of rCurrents
                    # Data are not valid inside rCurrents
                    if r0 >= rCurrents:
                        # Get divergence of B for integral
                        divB = calcDivB(batsrus, i, j, k, n, nI, nJ, nK, 
                                        dX, dY, dZ, _bx, _by, _bz)
                        
                        # To calculate the integral, we need the distance from 
                        # point i,j,k,n to XGSM
                        r[0] = XGSM[0] - batsrus.DataArray[_x,i,j,k,n]
                        r[1] = XGSM[1] - batsrus.DataArray[_y,i,j,k,n]
                        r[2] = XGSM[2] - batsrus.DataArray[_z,i,j,k,n]
                        rmag = np.sqrt( r[0]**2 + r[1]**2 + r[2]**2 )
                        
                        # dV for integral
                        measure = batsrus.DataArray[_measure,i,j,k,n]
                                    
                        ##########################################################
                        # Below we calculate the delta B in each differential volume 
                        # element in the integral.  We want the final result to be 
                        # in nT.
                        # dB = 1/(4pi) divB x r/r^3 dV
                        #    = 1/(4pi) [nT/Re] [Re] / [Re^3] * [Re^3]
                        #    = 1/(4pi) with distances in Re, B in nT
                        ##########################################################
                        
                        B = B + divB * r * measure / rmag**3 / 4 / np.pi 
      
    return B
  
def calc_ms_divBint_b(XGSM, timeISO, batsrus, rCurrents):
    """Process data in BATSRUS file to calculate the delta B at point XGSM.
    Helmholtz decomposition theorem used to convert Biot-Savart Law to a 
    surface integral used for calculation.  We will integrate across the outer
    boundary of the BATSRUS grid.  
    
    Inputs:
        XGSM = GSM (cartesian) position where magnetic field will be measured.
        
        timeISO = ISO time for data in BATSRUS file
              
        batsrus = BATSRUS data from swmfio
        
        rCurrents = range from earth center below which results are not valid.
            Measured in Re units.  We drop the data inside radius rCurrents

    Outputs:
        Bn, Be, Bd = cumulative sum of dB data in north-east-down coordinates,
            provides total B at point X (in SM coordinates)

        B = total B due to field-aligned currents (in SM coordinates)
        
    """

    logging.info(r'Calculate magnetosphere divB integral dB...')

    # We need the time to switch from GSM to SM coordinates
    time = iso2ints( timeISO )

    # Results in GSM coordinates
    BGSM = np.zeros(3)

    # Do volume integral
    BGSM = calc_ms_divBint_b_sub(XGSM, timeISO, batsrus, rCurrents)

    # Convert to SM coordinates        
    B = np.zeros(3)
    B = GSMtoSM( BGSM, time, ctype_in='car', ctype_out='car')
    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')
    
    # Get north, east, down components
    Bn, Be, Bd = get_NED_components( B, XSM )
     
    return Bn, Be, Bd, B[0], B[1], B[2]      


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

def loop_ms_divBint_b(info, point, reduce, deltahr=None, maxcores=20, deltaBlist=False):
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

        logging.info(f'Calculate magnetosphere divB integral dB for... {base}')
        
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
        Bn, Be, Bd, Bx, By, Bz = calc_ms_divBint_b(X, timeISO, batsrus, info['rCurrents'])
        
        return Bn, Be, Bd, Bx, By, Bz, Btime

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
                'Bx': Bx, 'By': By, 'Bz': Bz, 
                r'Time (hr)': Btimes, r'Datetime': dtimes,
                r'Month': dtimes_m, r'Day': dtimes_d,
                r'Hour': dtimes_hh, r'Minute': dtimes_mm}, index=dtimes)
    create_directory(info['dir_derived'], 'timeseries')
    if SECOND_ORDER:
        if USE_B1:
            pklname = 'dB_divB_msph_2nd_b1-' + point + '.pkl'
        else:
            pklname = 'dB_divB_msph_2nd_b-' + point + '.pkl'
    else:
        if USE_B1:
            pklname = 'dB_divB_msph_swmfio_b1-' + point + '.pkl'
        else:
            pklname = 'dB_divB_msph_swmfio_b-' + point + '.pkl'
    df.to_pickle( os.path.join( info['dir_derived'], 'timeseries', pklname) )
    

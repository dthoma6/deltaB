#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 07:21:06 2024

@author: Dean Thomas
"""

import logging
import numpy as np

class OpenGGCM_interpolator():
    """Class to interpolate OpenGGCM results.  Uses linear interpolation.
    '''

    """
    def __init__(self, openggcm):
        """Initialize openggcm_interpolator class
            
        Inputs:
            openggcm = OpenGGCM_dataframe reading of OpenGGCM CDF file
                 
        Outputs:
            None
        """
        # logging.info('Initializing openggcm interpolator class') 

        # Store instance data
        self.openggcm = openggcm
        self.var_data = {} # data to be interpolated in GSM coordinates
        
        self.varidx = dict(self.openggcm.varidx)
        
        self.nI       = self.openggcm.nI
        self.nJ       = self.openggcm.nJ
        self.nK       = self.openggcm.nK
        
        self.xtickGSE = self.openggcm.xtickGSE    # ticks in GSE
        self.ytickGSE = self.openggcm.ytickGSE
        self.ztickGSE = self.openggcm.ztickGSE

        self.DataArray = self.openggcm.DataArray  # data in GSM

        return

    def register_variable(self, varname):
        '''Creates interpolator for the varname dataset'''

        # logging.info('Initializing openggcm interpolator variable') 
             
        # store varname data to be interpolated in dictionary
        self.var_data[varname] = self.openggcm.DataArray[self.varidx[varname],:,:,:] 

        return

    def transfrom_GSMtoGSE( self, B ):
        """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
        vector.  Used for coordinate GSM to GSE transformation.
        """
        A = self.openggcm.GSM_to_GSE
        C = np.zeros(3)
        C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
        C[2] = A[2,0]*B[0] + A[2,1]*B[1] + A[2,2]*B[2]
        return C

    def interpolator(self, xvecGSM, varnameGSM):
        '''Interpolator for the varname dataset.
        
        xvecGSE in GSM coordinates, varnameGSM data in GSM coordinates
        '''
        if not isinstance(xvecGSM, np.ndarray):
            xvecGSM = np.array(xvecGSM)
        
        XGSM, YGSM, ZGSM = xvecGSM.T  # xvec can be used like this
        if not isinstance(XGSM, np.ndarray):
            XGSM = np.array([XGSM])
            YGSM = np.array([YGSM])
            ZGSM = np.array([ZGSM])

        # Make sure grid is in the expected order, see if-thens below
        assert self.xtickGSE[0] > self.xtickGSE[1]
        assert self.ytickGSE[0] > self.ytickGSE[1]
        assert self.ztickGSE[0] < self.ztickGSE[1]
           
        # Storage for results
        resultsGSM = np.zeros(len(XGSM))
    
        # Loop through input points for the interpolation
        for m in range(len(XGSM)):
            # We need XGSM, YGSM, ZGSM in GSE coordinates to simplify
            # the trilinear interpolation using original OpenGGCM GSE data
            xGSE = self.transfrom_GSMtoGSE( np.array([XGSM[m], YGSM[m], ZGSM[m]]) )
            
            # Find where point lies inside x,y,z grid (GSE)
            # We compare the xGSE point to the ticks along
            # the x,y,z axes in GSE coordinates.
            
            # Start at pos. end, march to neg. end.
            for i in range(self.nI-1):
                if self.xtickGSE[i+1] <= xGSE[0]: break

            # Start at pos. end, march to neg. end.          
            for j in range(self.nJ-1):
                if self.ytickGSE[j+1] <= xGSE[1]: break
            
            # Start at NEG. end, march to POS. end.
            # This is the opposite of x and y
            for k in range(self.nK-1):
                if self.ztickGSE[k+1] >= xGSE[2]: break
            
            # Now that we know i,j,k, we can complete the interpolation
            
            # Grid spacing in the cell that contains the point xGSE
            dxGSE = self.xtickGSE[i+1] - self.xtickGSE[i]
            dyGSE = self.ytickGSE[j+1] - self.ytickGSE[j]
            dzGSE = self.ztickGSE[k+1] - self.ztickGSE[k]
 
            # How far is xGSE from the grid point i,j,k?
            xdGSE = (xGSE[0] - self.xtickGSE[i])/dxGSE
            ydGSE = (xGSE[1] - self.ytickGSE[j])/dyGSE
            zdGSE = (xGSE[2] - self.ztickGSE[k])/dzGSE
 
            # https://en.wikipedia.org/wiki/Trilinear_interpolation
            c000 = self.var_data[varnameGSM][ i  ,j  ,k   ] 
            c001 = self.var_data[varnameGSM][ i  ,j  ,k+1 ] 
            c010 = self.var_data[varnameGSM][ i  ,j+1,k   ] 
            c100 = self.var_data[varnameGSM][ i+1,j  ,k   ] 
            c011 = self.var_data[varnameGSM][ i,j+1  ,k+1 ] 
            c110 = self.var_data[varnameGSM][ i+1,j+1,k   ] 
            c101 = self.var_data[varnameGSM][ i+1,j  ,k+1 ] 
            c111 = self.var_data[varnameGSM][ i+1,j+1,k+1 ] 

            c00 = c000*(1.-xdGSE) + c100*xdGSE
            c01 = c001*(1.-xdGSE) + c101*xdGSE
            c10 = c010*(1.-xdGSE) + c110*xdGSE
            c11 = c011*(1.-xdGSE) + c111*xdGSE

            c0 = c00*(1.-ydGSE) + c10*ydGSE
            c1 = c01*(1.-ydGSE) + c11*ydGSE

            resultsGSM[m] = c0*(1.-zdGSE) + c1*zdGSE

        return list(resultsGSM)

if __name__ == "__main__":
    
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.035400.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived'
    
    from deltaB.OpenGGCM_dataframe import get_openggcm_data_from_cdf

    # Test interpolation algorithm
    
    ogcmdata = get_openggcm_data_from_cdf(file)
    ogcm_interp = OpenGGCM_interpolator(ogcmdata)
    ogcm_interp.register_variable( 'bx' )
    
    # Get OpenGGCM x,y,z data   
    x_ = ogcmdata.varidx['x']
    y_ = ogcmdata.varidx['y']
    z_ = ogcmdata.varidx['z']
    
    ogcmdata.data_arr[ :, x_ ]
    ogcmdata.data_arr[ :, y_ ]
    ogcmdata.data_arr[ :, z_ ]

    # Pick a random point on simulation grid
    from random import randint
    
    nI = ogcmdata.nI
    nJ = ogcmdata.nJ
    nK = ogcmdata.nK
    
    i = randint(1,nI-1)
    j = randint(1,nJ-1)
    k = randint(1,nK-1)
    
    # Interpolate at the point on the grid.  Difference should be zero.
    x0 = ogcmdata.DataArray[ x_,i,j,k ]
    y0 = ogcmdata.DataArray[ y_,i,j,k ]
    z0 = ogcmdata.DataArray[ z_,i,j,k ]
    print( 'Test at sim grid pt: ', x0,y0,z0 )
    bx0 = ogcmdata.DataArray[ ogcmdata.varidx['bx'], i,j,k ]
    bx = ogcm_interp.interpolator( (x0,y0,z0), 'bx')[0]
    print( 'bx: ', bx0, 'bx diff: ', bx0-bx )
    
    # Interpolate at mid-point between current point and next point along x axis (GSE)
    x1 = ogcmdata.DataArray[ x_,i+1,j,k ]
    y1 = ogcmdata.DataArray[ y_,i+1,j,k ]
    z1 = ogcmdata.DataArray[ z_,i+1,j,k ]
    print( 'Test at x mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = ogcmdata.DataArray[ ogcmdata.varidx['bx'], i+1,j,k ]
    bxm = 0.5*(bx0 + bx1)
    bx = ogcm_interp.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Interpolate at mid-point between current point and next point along y axis (GSE)
    x1 = ogcmdata.DataArray[ x_,i,j+1,k ]
    y1 = ogcmdata.DataArray[ y_,i,j+1,k ]
    z1 = ogcmdata.DataArray[ z_,i,j+1,k ]
    print( 'Test at y mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = ogcmdata.DataArray[ ogcmdata.varidx['bx'], i,j+1,k ]
    bxm = 0.5*(bx0 + bx1)
    bx = ogcm_interp.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Interpolate at mid-point between current point and next point along z axis (GSE)
    x1 = ogcmdata.DataArray[ x_,i,j,k+1 ]
    y1 = ogcmdata.DataArray[ y_,i,j,k+1 ]
    z1 = ogcmdata.DataArray[ z_,i,j,k+1 ]
    print( 'Test at z mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = ogcmdata.DataArray[ ogcmdata.varidx['bx'], i,j,k+1 ]
    bxm = 0.5*(bx0 + bx1)
    bx = ogcm_interp.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )


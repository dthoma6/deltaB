#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 07:21:06 2024

@author: Dean Thomas
"""

import logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator

class OpenGGCM_interpolator2():
    """Class to interpolate OpenGGCM results.  Uses SciPy linear interpolator.
    '''

    """
    def __init__(self, openggcm, GSMIN=True):
        """Initialize openggcm_interpolator class
            
        Inputs:
            openggcm = OpenGGCM_dataframe reading of OpenGGCM CDF file
                 
            GSMIN = True => GSM coordinates provided for interpolation grid,
                    False => GSE coordinates provided
        Outputs:
            None
        """
        # logging.info('Initializing openggcm interpolator class') 

        # Store instance data
        self.openggcm = openggcm
        self.var_interp = {} # interpolator in GSE coordinates but GSM var data
        
        self.varidx = dict(self.openggcm.varidx)
        
        self.nI       = self.openggcm.nI
        self.nJ       = self.openggcm.nJ
        self.nK       = self.openggcm.nK
        
        self.xtickGSE = self.openggcm.xtickGSE    # ticks in GSE
        self.ytickGSE = self.openggcm.ytickGSE
        self.ztickGSE = self.openggcm.ztickGSE
        
        self.DataArray = self.openggcm.DataArray  # data in GSM

        self.GSMIN = GSMIN

        return

    def register_variable(self, varname):
        '''Creates interpolator for the varname dataset'''

        # logging.info('Initializing openggcm interpolator variable') 
             
        # store varname interpolator in dictionary
        data = self.openggcm.DataArray[self.varidx[varname],:,:,:] # GSM 
        x = self.xtickGSE
        y = self.ytickGSE
        z = self.ztickGSE
        self.var_interp[varname] = RegularGridInterpolator((x, y, z), data)

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
            # We need XGSM, YGSM, ZGSM in GSE coordinates for interpolation
            # Coordinates are in GSE, but data are in GSM

            # Note: GSMIN tells us whether to expect GSM or GSE coordinates as input
            if self.GSMIN: 
                xGSE = self.transfrom_GSMtoGSE( np.array([XGSM[m], YGSM[m], ZGSM[m]]) )
            else:
                xGSE = np.array([XGSM[m], YGSM[m], ZGSM[m]])
            
            # Interpolate
            tmp = self.var_interp[varnameGSM]( xGSE )
            if tmp.ndim > 0:
                resultsGSM[m] = self.var_interp[varnameGSM]( xGSE )[0]
            else:
                resultsGSM[m] = self.var_interp[varnameGSM]( xGSE )

        return list(resultsGSM)

if __name__ == "__main__":
    
    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_1/GM_CDF/Dean_Thomas_052924_1.3df.035400.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_1.derived'
    
    from deltaB.OpenGGCM_dataframe import get_openggcm_data_from_cdf

    # Test interpolation algorithm
    
    ogcmdata = get_openggcm_data_from_cdf(file)

    from deltaB import OpenGGCM_interpolator
    ogcm_interp = OpenGGCM_interpolator(ogcmdata)
    ogcm_interp.register_variable( 'bx' )

    ogcm_interp2 = OpenGGCM_interpolator2(ogcmdata)
    ogcm_interp2.register_variable( 'bx' )
    
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
    bx = ogcm_interp2.interpolator( (x0,y0,z0), 'bx')[0]
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
    bx = ogcm_interp2.interpolator( (xm,ym,zm), 'bx')[0]
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
    bx = ogcm_interp2.interpolator( (xm,ym,zm), 'bx')[0]
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
    bx = ogcm_interp2.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Compare results to other OpenGGCM interpolator        
    
    NUM = 10000
    bx1 = np.zeros(NUM)
    bx2 = np.zeros(NUM)
    
    for i in range(NUM):
        i1 = randint(0,nI-1)
        j1 = randint(0,nJ-1)
        k1 = randint(0,nK-1)
        i2 = randint(0,nI-1)
        j2 = randint(0,nJ-1)
        k2 = randint(0,nK-1)

        x1 = ogcmdata.DataArray[ x_,i1,j1,k1 ]
        y1 = ogcmdata.DataArray[ y_,i1,j1,k1 ]
        z1 = ogcmdata.DataArray[ z_,i1,j1,k1 ]
        x2 = ogcmdata.DataArray[ x_,i2,j2,k2 ]
        y2 = ogcmdata.DataArray[ y_,i2,j2,k2 ]
        z2 = ogcmdata.DataArray[ z_,i2,j2,k2 ]

        x0 = 0.5*(x1+x2)
        y0 = 0.5*(y1+y2)
        z0 = 0.5*(z1+z2)
        
        r = np.sqrt( x0**2 + y0**2 + z0**2 )
        
        if r > ogcmdata.rCurrents:
            bx2[i] = ogcm_interp2.interpolator( (x0,y0,z0), 'bx')[0]
            bx1[i] = ogcm_interp.interpolator( (x0,y0,z0), 'bx')[0]
            
    import matplotlib.pyplot as plt
    
    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Linear bx')
    plt.ylabel('SciPy bx')
    plt.show()

    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Linear bx')
    plt.ylabel('SciPy bx')
    plt.xlim((-50,50))
    plt.ylim((-50,50))
    plt.show()


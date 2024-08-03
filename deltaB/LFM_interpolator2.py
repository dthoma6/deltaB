#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 07:21:06 2024

@author: Dean Thomas
"""

import logging
import numpy as np

# Number of nearest neighbors to include in interpolation
NEIGHBORS = 8

# Power used in inverse distance weighting interpolation
# Default is 2
POWER = 2

class LFM_interpolator2():
    """Class tp create inverse distance weighted interpolator for LFM results.

    """
    def __init__(self, lfm):
        """Initialize lfm_interpolator class
            
        Inputs:
            lfm = LFM_dataframe reading of LFM CDF file
                 
        Outputs:
            None
        """
        logging.info('Initializing lfm interpolator class') 

        # Store instance data
        self.lfm = lfm
        self.data = {}                       # data for interpolator in GSM coordinates
        
        self.varidx = dict(self.lfm.varidx)
        
        self.DataArray = self.lfm.DataArray  # data in GSM
        
        self.xcenterSM  = self.lfm.xcenterSM # cell centers SM cylindrical coordinates
        self.rcenterSM  = self.lfm.rcenterSM   
        self.acenterSM  = self.lfm.acenterSM   
        
        return
    
    def register_variable(self, varname):
        '''Creates interpolator for the varname dataset'''

        logging.info('Initializing lfm interpolator variable') 
        
        # store data for varname interpolator
        self.data[varname] = self.DataArray[self.varidx[varname],:,:,:]

        return

    def transfrom_GSMtoSM( self, B ):
        """Matrix multiplication of A (3x3) matrix with B (3) vector to give C (3)
        vector.  Used for coordinate GSM to SM transformation.
        """
        # logging.info('lfm GSM to SM transformation') 

        A = self.lfm.GSM_to_SM
        C = np.zeros(3)
        C[0] = A[0,0]*B[0] + A[0,1]*B[1] + A[0,2]*B[2]
        C[1] = A[1,0]*B[0] + A[1,1]*B[1] + A[1,2]*B[2]
        C[2] = A[2,0]*B[0] + A[2,1]*B[1] + A[2,2]*B[2]
        return C

    def interpolator(self, xvecGSM, varname):
        '''Interpolator for the varname dataset.
        
        xvecGSM in GSM coordinates, varname data in GSM coordinates
        '''
        # logging.info('lfm interpolation') 

        if not isinstance(xvecGSM, np.ndarray):
            xvecGSM = np.array(xvecGSM)
        
        XGSM, YGSM, ZGSM = xvecGSM.T  # xvec can be used like this
        if not isinstance(XGSM, np.ndarray):
            XGSM = np.array([XGSM])
            YGSM = np.array([YGSM])
            ZGSM = np.array([ZGSM])
        
        # array for results
        resultsGSM = np.zeros( len(XGSM) )
        
        # Loop thru GSM coordinates 
        for m in range(len(XGSM)):
            
            # Find the point that we interpolate to in SM cylindrical coordinates
            # We use the fact that the LFM data is in sheets of constant azimuth
            xSM = self.transfrom_GSMtoSM(np.array([XGSM[m], YGSM[m], ZGSM[m]]))
            
            rSM = np.sqrt( xSM[1]**2 + xSM[2]**2 )   
            azSM = np.arctan2( xSM[2], xSM[1] )
            # We want az in the range 0-> 2pi
            if azSM < 0: azSM = 2*np.pi + azSM

            # Find index of nearest neighbors on slice
            d2 = (self.xcenterSM - xSM[0])**2 + (self.rcenterSM - rSM)**2
            idx = np.unravel_index(np.argsort(d2, axis=None), d2.shape)
            
            # Find which azimuth sheets the point xSM lies between
            for k in range( len(self.acenterSM) ):
                if self.acenterSM[k] > azSM: break
            assert k > 0
    
            # Data in GSM coordinates
            data0 = self.data[varname][:,:,k  ]      
            data1 = self.data[varname][:,:,k-1]     
            
            # Get data for nearest neighbor. v0 and v1 are for
            # the azimuth slices bracketing the point XGSM in SM
            v0 = data0[idx[0][0], idx[1][0]]
            v1 = data1[idx[0][0], idx[1][0]]
            x0 = self.xcenterSM[idx[0][0], idx[1][0]]
            r0 = self.rcenterSM[idx[0][0], idx[1][0]]
            d0 = np.sqrt( (xSM[0]-x0)**2 + (rSM-r0)**2 )
            
            # Stop here if the point XGSM in SM is on a grid point
            # No interpolation needed
            if np.isclose( d0, 0., atol=1e-5 ): 
               # Interpolate between azimuth slices
               daz = self.acenterSM[k] - self.acenterSM[k-1]
               dazSM = azSM - self.acenterSM[k-1]
               resultsGSM[m] = v1 + (v0-v1)*dazSM/daz
               
            # If not on a grid point, we interpolate
            # Since we don't have a regular grid, we use
            # inverse distance weighted interpolation
            # https://en.wikipedia.org/wiki/Inverse_distance_weighting
            else:
                # Use data from nearest neighbor
                w = 1/d0**POWER
                vv0 = v0/d0**POWER
                vv1 = v1/d0**POWER
                
                # Add data from the next NEIGHBORS-1 nearest neighbors
                for i in range(1,NEIGHBORS):
                
                    v0 = data0[idx[0][i], idx[1][i]]
                    v1 = data1[idx[0][i], idx[1][i]]
                    
                    x0 = self.xcenterSM[idx[0][i], idx[1][i]]
                    r0 = self.rcenterSM[idx[0][i], idx[1][i]]
                    
                    d0 = np.sqrt( (xSM[0]-x0)**2 + (rSM-r0)**2 )

                    vv0 += v0/d0**POWER
                    vv1 += v1/d0**POWER
                    w += 1/d0**POWER
                
                # Finish inverse distance weighting interpolation
                vv0 = vv0/w
                vv1 = vv1/w
                
                # Linear interpolation between azimuth slices
                daz = self.acenterSM[k] - self.acenterSM[k-1]
                dazSM = azSM - self.acenterSM[k-1]
                resultsGSM[m] = vv1 + (vv0-vv1)*dazSM/daz
                
        return list(resultsGSM)

if __name__ == "__main__":
    
    from deltaB.LFM_dataframe import get_lfm_data_from_cdf

    file = '/Volumes/PhysicsHD/Dean_Thomas_052924_2/GM_CDF/null.LFM.Dean_Thomas_052924_2_mhd_2000-01-01T02-20-00Z.cdf'
    dir_derived = '/Volumes/PhysicsHD/Dean_Thomas_052924_2.derived'

    from datetime import datetime
    now = datetime.now()
    print('Start: ', now.time())
    

    lfmdata = get_lfm_data_from_cdf(file)

    end = datetime.now()
    print('Finish: ', end.time())

    from deltaB import LFM_interpolator
    lfm_interp = LFM_interpolator(lfmdata)
    lfm_interp.register_variable( 'bx' )
    
    lfm_interp2 = LFM_interpolator2(lfmdata)
    lfm_interp2.register_variable( 'bx' )
    
    # Pick a random point on simulation grid
    from random import randint
    
    nI = lfmdata.nI
    nJ = lfmdata.nJ
    nK = lfmdata.nK
    
    i = randint(1,nI-1)
    j = randint(1,nJ-1)
    k = randint(1,nK-1)
        
    # Get LFM x,y,z data   
    x_ = lfmdata.varidx['x']
    y_ = lfmdata.varidx['y']
    z_ = lfmdata.varidx['z']
    
    # Interpolate at the point on the grid.  Difference should be zero.
    x0 = lfmdata.DataArray[ x_,i,j,k ]
    y0 = lfmdata.DataArray[ y_,i,j,k ]
    z0 = lfmdata.DataArray[ z_,i,j,k ]
    print( 'Test at sim grid pt: ', x0,y0,z0 )
    bx0 = lfmdata.DataArray[ lfmdata.varidx['bx'], i,j,k ]
    bx = lfm_interp2.interpolator( (x0,y0,z0), 'bx')[0]
    bx1 = lfm_interp.interpolator( (x0,y0,z0), 'bx')[0]
    print( 'bx: ', bx0, bx1, 'bx frac diff: ', (bx0-bx)/bx0 )
    
    # Compare results to baryocentric interpolator in LFM_interpolator        
     
    NUM = 10000
    bx1 = np.zeros(NUM)
    bx2 = np.zeros(NUM)
    
    for i in range(NUM):
        i1 = randint(0,nI)
        j1 = randint(0,nJ)
        k1 = randint(0,nK)
        i2 = randint(0,nI)
        j2 = randint(0,nJ)
        k2 = randint(0,nK)

        x1 = lfmdata.DataArray[ x_,i1,j1,k1 ]
        y1 = lfmdata.DataArray[ y_,i1,j1,k1 ]
        z1 = lfmdata.DataArray[ z_,i1,j1,k1 ]
        x2 = lfmdata.DataArray[ x_,i2,j2,k2 ]
        y2 = lfmdata.DataArray[ y_,i2,j2,k2 ]
        z2 = lfmdata.DataArray[ z_,i2,j2,k2 ]

        x0 = 0.5*(x1+x2)
        y0 = 0.5*(y1+y2)
        z0 = 0.5*(z1+z2)
        
        r = np.sqrt( x0**2 + y0**2 + z0**2 )
        
        if r > lfmdata.rCurrents:
            bx2[i] = lfm_interp2.interpolator( (x0,y0,z0), 'bx')[0]
            bx1[i] = lfm_interp.interpolator( (x0,y0,z0), 'bx')[0]
            
    import matplotlib.pyplot as plt
    
    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Distance Weighted bx')
    plt.ylabel('Baryocentric bx')
    plt.show()

    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Distance Weighted bx')
    plt.ylabel('Baryocentric bx')
    plt.xlim((-50,50))
    plt.ylim((-50,50))
    plt.show()

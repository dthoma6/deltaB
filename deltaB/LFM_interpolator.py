#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 07:21:06 2024

@author: Dean Thomas
"""

import logging
import numpy as np
                
class LFM_interpolator():
    """Class to create baryocentric interpolator for LFM results.  Defaults to
    nearest neighbor interpolation outside volume of data points.
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
        
        self.nI = lfm.nI
        self.nJ = lfm.nJ
        self.nK = lfm.nK
        
        self.varidx = dict(self.lfm.varidx)
        
        self.DataArray = self.lfm.DataArray  # data in GSM
        
        self.xsliceSM  = self.lfm.xsliceSM   # cell centers SM cylindrical coordinates
        self.rsliceSM  = self.lfm.rsliceSM   
        self.asliceSM  = self.lfm.asliceSM   
        
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

    def check_triangle(self, i1, i2, data0, data1, xsliceSM, rsliceSM, xSM, rSM, x1, r1):
        """Function used by interpolator.  Contains code to check whether
        point is inside triangle defined by grid points.  Uses baryocentric
        grid for interpolation.  If l1, l2, l3 >=0 and sum to one, (xSM[0],rSM) 
        is in the triangle defined by (x1,r1), (x2,r2), and (x3,r3)
        """
        
        # Get data on values at (x2,r2), and (x3,r3).  Calling routine
        # already has values at (x1,r1)
        
        v20 = data0[i1]
        v21 = data1[i1]
        x2  = xsliceSM[i1]
        r2  = rsliceSM[i1]

        v30 = data0[i2]
        v31 = data1[i2]
        x3  = xsliceSM[i2]
        r3  = rsliceSM[i2]
        
        # See https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        # for discussion of baryocentric interpolation and calculation of
        # l1, l2, and l3

        l1 = ( (r2 - r3)*(xSM[0] - x3) + (x3 - x2)*(rSM - r3) ) / \
             ( (r2 - r3)*(x1     - x3) + (x3 - x2)*(r1  - r3) ) 
             
        l2 = ( (r3 - r1)*(xSM[0] - x3) + (x1 - x3)*(rSM - r3) ) / \
             ( (r2 - r3)*(x1     - x3) + (x3 - x2)*(r1  - r3) )   
             
        l3 = 1 - l1 - l2
        
        # If true, point is in triangle
        found = l1 >= 0 and l2 >=0 and l3 >= 0
        
        return v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found

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

            # Find index (i,j) of the nearest neighbor on slice
            # We then look at triangles of points around (i,j) to find
            # which triangle contains xSM
            dist = (self.xsliceSM - xSM[0])**2 + (self.rsliceSM - rSM)**2
            idx = np.unravel_index(np.argsort(dist, axis=None), dist.shape)
            i = idx[0][0]
            j = idx[1][0]
            
            # Find which azimuth sheets the point xSM lies between
            for k in range( len(self.asliceSM) ):
                if self.asliceSM[k] > azSM: break
            
            # Worry about wrap around in azimuth
            if k > 0:
                k2 = k - 1
            else:
                k2 = self.asliceSM.shape[0] - 1
    
            # Data in SM coordinates, used repeatedly below
            data0 = self.data[varname][:,:,k  ]      
            data1 = self.data[varname][:,:,k2 ]    
            xsliceSM = self.xsliceSM
            rsliceSM = self.rsliceSM
            
            # Get data for nearest neighbor. v00 and v01 are data for
            # the azimuth slices bracketing the point xSM. We'll 
            # interpolate between the values at the end.
            idx1 = (i, j)
            v10 = data0[idx1]
            v11 = data1[idx1]
            x1  = self.xsliceSM[idx1]
            r1  = self.rsliceSM[idx1]
            d1  = np.sqrt( (xSM[0]-x1)**2 + (rSM-r1)**2 )
            
            # Stop here if the point XGSM in SM is on a grid point
            # No interpolation needed
            if np.isclose( d1, 0., atol=1e-5 ): 
               # Interpolate between azimuth slices
               if k > 0:  # Worry about wrap around in azimuth
                   daz = self.asliceSM[k] - self.asliceSM[k2]
                   dazSM = azSM - self.asliceSM[k2]
               else:
                   daz = 2*np.pi + self.asliceSM[k] - self.asliceSM[k2]
                   dazSM = 2*np.pi + azSM - self.asliceSM[k2]                   
               resultsGSM[m] = v11 + (v10-v11)*dazSM/daz
               
            # If not on a grid point, we interpolate
            # Since we don't have a regular grid, we use baryocentric interpolation
            # https://en.wikipedia.org/wiki/Barycentric_coordinate_system
            # See interpolation on triangular unstructured grid
            else:
                # Find indices for points around idx1 that we'll explore
                # We'll form triangles with idx1 plus pairs of the points below
                idx2  = (i    , j + 1)
                idx3  = (i + 1, j + 1)
                idx4  = (i + 1, j    )
                idx5  = (i + 1, j - 1)
                idx6  = (i    , j - 1)
                idx7  = (i - 1, j - 1)
                idx8  = (i - 1, j    )
                idx9  = (i - 1, j + 1)
                
                idx10 = (i    , j + 2)
                idx11 = (i + 1, j + 2)
                idx12 = (i + 1, j - 2)
                idx13 = (i    , j - 2)
                idx14 = (i - 1, j - 2)
                idx15 = (i - 1, j + 2)
                
                idx16 = (i    , j + 3)
                idx17 = (i + 1, j + 3)
                idx18 = (i + 1, j - 3)
                idx19 = (i    , j - 3)
                idx20 = (i - 1, j - 3)
                idx21 = (i - 1, j + 3)

                idx22 = (i + 2, j + 1)
                idx23 = (i + 2, j    )
                idx24 = (i + 2, j - 1)
                idx25 = (i - 2, j - 1)
                idx26 = (i - 2, j    )
                idx27 = (i - 2, j + 1)
                    
                # True when we have found which triangle that contains xSM
                found = False
                
                # Triangles immediately around nearest neighbor in i,j plane
                                    
                # Triangle A
                if not found and i < self.nI-1 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx2, idx3, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle B
                if not found and i < self.nI-1 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx3, idx4, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle C
                if not found and i < self.nI-1 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx4, idx5, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle D
                if not found and i < self.nI-1 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx5, idx6, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle E
                if not found and i > 0 and j > 0 :
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx6, idx7, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
 
                # Triangle F
                if not found and i > 0 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx7, idx8, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
 
                # Triangle G
                if not found and i > 0 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx8, idx9, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
  
                # Triangle H
                if not found and i > 0 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx9, idx2, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangles further out along j, needed because of asymmetry
                # in distorted spherical coordinate grid spacing
                        
                # Triangle I
                if not found and i < self.nI-1 and j < self.nJ-2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx10, idx11, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle J
                if not found and i < self.nI-1 and j < self.nJ-2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx11, idx3, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle K
                if not found and i < self.nI-1 and j > 1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx5, idx12, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
     
                # Triangle L
                if not found and i < self.nI-1 and j > 1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx12, idx13, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
   
                # Triangle M
                if not found and i > 0 and j > 1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx13, idx14, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
   
                # Triangle N
                if not found and i > 0 and j > 1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx14, idx7, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle O
                if not found and i < self.nI-1 and j < self.nJ-2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx9, idx15, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                        
                # Triangle P
                if not found and i < self.nI-1 and j < self.nJ-2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx15, idx10, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangles even further out along j, needed because of asymmetry
                # in distorted spherical coordinate grid spacing

                # Triangle Q
                if not found and i < self.nI-1 and j < self.nJ-3:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx16, idx17, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle R
                if not found and i < self.nI-1 and j < self.nJ-3:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx17, idx11, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle S
                if not found and i < self.nI-1 and j > 2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx12, idx18, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
     
                # Triangle T
                if not found and i < self.nI-1 and j > 2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx18, idx19, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
   
                # Triangle U
                if not found and i > 0 and j > 2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx19, idx20, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
   
                # Triangle V
                if not found and i > 0 and j > 2:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx20, idx14, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Triangle W
                if not found and i < self.nI-1 and j < self.nJ-3:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx15, idx21, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                        
                # Triangle X
                if not found and i < self.nI-1 and j < self.nJ-3:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx21, idx16, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                        
                # No triangles Y or Z

                # Triangles further out along i, needed because of asymmetry
                # in distorted spherical coordinate grid spacing

                # Triangle AA
                if not found and i < self.nI-3 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx3, idx22, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle BB
                if not found and i < self.nI-3 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx22, idx23, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle CC
                if not found and i < self.nI-3 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx23, idx24, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle DD
                if not found and i < self.nI-3 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx24, idx5, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle EE
                if not found and i > 1 and j > 0 :
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx7, idx25, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle FF
                if not found and i > 1 and j > 0:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx25, idx26, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                
                # Triangle GG
                if not found and i > 1 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx26, idx27, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)
                # Triangle HH
                if not found and i > 1 and j < self.nJ-1:
                    v20, v21, x2, r2, v30, v31, x3, r3, l1, l2, l3, found = \
                        self.check_triangle(idx27, idx9, data0, data1, xsliceSM, 
                                            rsliceSM, xSM, rSM, x1, r1)

                # Warn if we could not find triangle
                if not found:
                    # We ignore warning for common edge cases
                    # These are outside the volume of grid points
                    if j!= 0 and j != self.nJ-1 and i != 0 and i!= self.nI-1:
                        logging.warning(f'Default to nearest neighbor interpolation, triangle not found at {xSM} {i} {j} {k}')

                    # Interpolate between nearest neighbors on azimuth sheets 
                    if k > 0:  # Worry about wrap around in azimuth
                        daz = self.asliceSM[k] - self.asliceSM[k2]
                        dazSM = azSM - self.asliceSM[k2]
                    else:
                        daz = 2*np.pi + self.asliceSM[k] - self.asliceSM[k2]
                        dazSM = 2*np.pi + azSM - self.asliceSM[k2]                   
                    resultsGSM[m] = v11 + (v10-v11)*dazSM/daz
                else:    
                    # Otherwise finish baryocentric interpolation
                    vv0 = v10 * l1 + v20 * l2 + v30 * l3
                    vv1 = v11 * l1 + v21 * l2 + v31 * l3
                    
                    # Linear interpolation between azimuth slices
                    if k > 0: # Worry about wrap around in azimuth
                        daz = self.asliceSM[k] - self.asliceSM[k2]
                        dazSM = azSM - self.asliceSM[k2]
                    else:
                        daz = 2*np.pi + self.asliceSM[k] - self.asliceSM[k2]
                        dazSM = 2*np.pi + azSM - self.asliceSM[k2]                   
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

    lfm_interp = LFM_interpolator(lfmdata)
    lfm_interp.register_variable( 'bx' )
    
    # Pick a random point on simulation grid
    from random import randint
    
    nI = lfmdata.nI
    nJ = lfmdata.nJ
    nK = lfmdata.nK
        
    # Get LFM x,y,z data   
    x_ = lfmdata.varidx['x']
    y_ = lfmdata.varidx['y']
    z_ = lfmdata.varidx['z']
        
    import random
    
    random.seed(21) #(15)
    
    for i in range(10000):
        i1 = randint(0,nI-1)        
        j1 = randint(0,nJ-1)
        k1 = randint(0,nK-1)
        i2 = randint(0,nI-1)
        j2 = randint(0,nJ-1)
        k2 = randint(0,nK-1)

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
            bx = lfm_interp.interpolator( (x0,y0,z0), 'bx')[0]
            # print( 'Test at pt: ', x0,y0,z0, ' bx: ', bx )
   
    i = randint(0,nI-1)
    j = randint(0,nJ-1)
    k = randint(0,nK-1)
    
    # Interpolate at the point on the grid.  Difference should be zero.
    x0 = lfmdata.DataArray[ x_,i,j,k ]
    y0 = lfmdata.DataArray[ y_,i,j,k ]
    z0 = lfmdata.DataArray[ z_,i,j,k ]
    print( 'Test at sim grid pt: ', x0,y0,z0 )
    
    bx0 = lfmdata.DataArray[ lfmdata.varidx['bx'], i,j,k ]    
    bx = lfm_interp.interpolator( (x0,y0,z0), 'bx')[0]
    
    print( 'bx: ', bx, 'bx frac diff: ', (bx0-bx)/bx0 )
    
 
        
    
    

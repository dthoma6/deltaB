#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:48:10 2024

@author: Dean Thomas
"""

import logging
import numpy as np
import numba

@numba.njit
def F2P(fortran_index):
    return fortran_index - 1

@numba.njit
def P2F(python_index):
    return python_index + 1

class BATSRUS_interpolator2():
    """Class to interpolate BATSRUS results.  Heavily-based on swmfio by Gary
    Quaresima.  
    '''

    """
    def __init__(self, batsrus):
        """Initialize batsrus_interpolator class
            
        Inputs:
            batsrus = BATSRUS_dataframe reading of BATSRUS file, contains SWMF results
                 
        Outputs:
            None
        """
        # logging.info('Initializing batsrus interpolator class') 

        # Store instance data
        self.batsrus = batsrus
        self.var_data = {}   # data to be interpolated in GSM coordinates
        
        self.varidx = dict(self.batsrus.varidx)

        return
    
    def find_tree_node(self, point):
        '''Finds which node that contains point '''

        xin = self.batsrus.xGlobalMin <= point[0] <= self.batsrus.xGlobalMax
        yin = self.batsrus.yGlobalMin <= point[1] <= self.batsrus.yGlobalMax
        zin = self.batsrus.zGlobalMin <= point[2] <= self.batsrus.zGlobalMax
        if not (xin and yin and zin): 
            raise RuntimeError('point out of simulation volume')

        found = False
        for iNode in self.batsrus.amr_level_0_nodes:
            minx = self.batsrus.block_x_min[F2P(iNode)]
            maxx = self.batsrus.block_x_max[F2P(iNode)]
            miny = self.batsrus.block_y_min[F2P(iNode)]
            maxy = self.batsrus.block_y_max[F2P(iNode)]
            minz = self.batsrus.block_z_min[F2P(iNode)]
            maxz = self.batsrus.block_z_max[F2P(iNode)]

            p1 = minx <= point[0] <= maxx
            p2 = miny <= point[1] <= maxy
            p3 = minz <= point[2] <= maxz
            if p1 and p2 and p3:
                found = True
                break

        assert(found == True)

        while True:
            if self.batsrus.block_child_count[F2P(iNode)] == 0:
                break

            found = False
            for j in range(self.batsrus.block_child_count[F2P(iNode)]):
                child = self.batsrus.block_child_ids[j, F2P(iNode)]

                xin = self.batsrus.block_x_min[F2P(child)] <= point[0] <= self.batsrus.block_x_max[F2P(child)]
                yin = self.batsrus.block_y_min[F2P(child)] <= point[1] <= self.batsrus.block_y_max[F2P(child)]
                zin = self.batsrus.block_z_min[F2P(child)] <= point[2] <= self.batsrus.block_z_max[F2P(child)]

                if xin and yin and zin:
                    found = True
                    iNode = child
                    break
            # TODO: Add check for max depth to prevent loop from never ending.
        return iNode

    def register_variable(self, varname):
        '''Stores varname data for interpolator'''

        # logging.info('Initializing batsrus interpolator variable') 

        # store varname data to be interpolated in dictionary
        self.var_data[varname] = self.batsrus.DataArray[self.varidx[varname],:,:,:,:] 

        return

    def interpolator(self, xvecGSM, varname):
        '''Interpolator for the varname dataset.'''

        if not isinstance(xvecGSM, np.ndarray):
            xvecGSM = np.array(xvecGSM)
        
        XGSM, YGSM, ZGSM = xvecGSM.T  # xvec can be used like this
        if not isinstance(XGSM, np.ndarray):
            XGSM = np.array([XGSM])
            YGSM = np.array([YGSM])
            ZGSM = np.array([ZGSM])

        _x = self.batsrus.varidx['x']
        _y = self.batsrus.varidx['y']
        _z = self.batsrus.varidx['z']
        
        DA = self.batsrus.DataArray
        
        nI = self.batsrus.nI
        nJ = self.batsrus.nJ
        nK = self.batsrus.nK

        # Storage for results
        resultsGSM = np.zeros(len(XGSM))
    
        # Loop through input points for the interpolation
        for m in range(len(XGSM)):
            point = np.array([XGSM[m], YGSM[m], ZGSM[m]])
            iNode = self.find_tree_node(point)
            iBlockP = self.batsrus.node2block[F2P(iNode)]
    
            # get the gridspacing in x,y,z
            gridspacingX = DA[_x,1,0,0,iBlockP] - DA[_x,0,0,0,iBlockP]
            gridspacingY = DA[_y,0,1,0,iBlockP] - DA[_y,0,0,0,iBlockP]
            gridspacingZ = DA[_z,0,0,1,iBlockP] - DA[_z,0,0,0,iBlockP]
    
            # i0 is s.t. the highest index s.t. the x coordinate of the 
            #  corresponding cell block_data[iNode,i0,:,:]  is still less than point[0]
            i0 = (point[0] - DA[_x, 0, 0, 0, iBlockP])/gridspacingX
            j0 = (point[1] - DA[_y, 0, 0, 0, iBlockP])/gridspacingY
            k0 = (point[2] - DA[_z, 0, 0, 0, iBlockP])/gridspacingZ
            i0 = int(np.floor(i0))
            j0 = int(np.floor(j0))
            k0 = int(np.floor(k0))
    
            # i1 = i0+1 is the lowest index s.t. the x coordinate of the 
            # corresponding cell block_data[iNode,i1,:,:]  is still greater than point[0]
            # together, i0 and i1 form the upper and lower bounds for a linear interpolation in x
            # likewise for j0,j1,y  and k0,k1,z
    
            if i0 == -1:
                i0 = 0
                i1 = 1
            elif i0 == nI-1:
                i0 = nI-2
                i1 = nI-1
            else:
                i1 = i0 + 1
    
            if j0 == -1:
                j0 = 0
                j1 = 1
            elif j0 == nJ-1:
                j0 = nJ-2
                j1 = nJ-1
            else:
                j1 = j0 + 1
    
            if k0 == -1:
                k0 = 0
                k1 = 1
            elif k0 == nK-1:
                k0 = nK-2
                k1 = nK-1
            else:
                k1 = k0 + 1
    
            # All together i0,i1,j0, etc... form a cube of side length "gridpacing"
            # To do trilinear interpolation within, define xd as the distance
            # along x of point within that cube, in units of "gridspacing"
            xd = (point[0] - DA[_x, i0, 0 , 0 , iBlockP])/gridspacingX
            yd = (point[1] - DA[_y, 0 , j0, 0 , iBlockP])/gridspacingY
            zd = (point[2] - DA[_z, 0 , 0 , k0, iBlockP])/gridspacingZ
            
            #https://en.wikipedia.org/wiki/Trilinear_interpolation
            c000 = self.var_data[varname][ i0, j0, k0,  iBlockP]
            c001 = self.var_data[varname][ i0, j0, k1,  iBlockP]
            c010 = self.var_data[varname][ i0, j1, k0,  iBlockP]
            c100 = self.var_data[varname][ i1, j0, k0,  iBlockP]
            c011 = self.var_data[varname][ i0, j1, k1,  iBlockP]
            c110 = self.var_data[varname][ i1, j1, k0,  iBlockP]
            c101 = self.var_data[varname][ i1, j0, k1,  iBlockP]
            c111 = self.var_data[varname][ i1, j1, k1,  iBlockP]
    
            c00 = c000*(1.-xd) + c100*xd
            c01 = c001*(1.-xd) + c101*xd
            c10 = c010*(1.-xd) + c110*xd
            c11 = c011*(1.-xd) + c111*xd
    
            c0 = c00*(1.-yd) + c10*yd
            c1 = c01*(1.-yd) + c11*yd
    
            resultsGSM[m] = c0*(1.-zd) + c1*zd
            
        return resultsGSM

if __name__ == "__main__":
    
    file = '/Volumes/PhysicsHD/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-193800-000.out.cdf'
    dir_derived = '/Volumes/PhysicsHD/Bob_Weigel_070323_3.derived'
    info = {} # empty info dict
   
    from deltaB.BATSRUS_dataframe import get_batsrus_data_from_cdf

    # Test interpolation algorithm
    batsdata = get_batsrus_data_from_cdf(file,info)

    from deltaB import BATSRUS_interpolator
    bats_interp = BATSRUS_interpolator(batsdata)
    bats_interp.register_variable( 'bx' )

    bats_interp2 = BATSRUS_interpolator2(batsdata)
    bats_interp2.register_variable( 'bx' )
    
    # Get OpenGGCM x,y,z data   
    x_ = batsdata.varidx['x']
    y_ = batsdata.varidx['y']
    z_ = batsdata.varidx['z']
    
    batsdata.data_arr[ :, x_ ]
    batsdata.data_arr[ :, y_ ]
    batsdata.data_arr[ :, z_ ]

    # Pick a random point on simulation grid
    from random import randint
    
    nI = batsdata.nI
    nJ = batsdata.nJ
    nK = batsdata.nK
    nBlock = batsdata.nBlock
    
    i = randint(0,nI-2)
    j = randint(0,nJ-2)
    k = randint(0,nK-2)
    n = randint(0,nBlock-1)
    
    # Interpolate at the point on the grid.  Difference should be zero.
    x0 = batsdata.DataArray[ x_,i,j,k,n ]
    y0 = batsdata.DataArray[ y_,i,j,k,n ]
    z0 = batsdata.DataArray[ z_,i,j,k,n ]
    print( 'Test at sim grid pt: ', x0,y0,z0 )
    bx0 = batsdata.DataArray[ batsdata.varidx['bx'], i,j,k,n ]
    bx = bats_interp2.interpolator( (x0,y0,z0), 'bx')[0]
    print( 'bx: ', bx0, 'bx diff: ', bx0-bx )
    
    # Interpolate at mid-point between current point and next point along x axis (GSE)
    x1 = batsdata.DataArray[ x_,i+1,j,k,n ]
    y1 = batsdata.DataArray[ y_,i+1,j,k,n ]
    z1 = batsdata.DataArray[ z_,i+1,j,k,n ]
    print( 'Test at x mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = batsdata.DataArray[ batsdata.varidx['bx'], i+1,j,k,n ]
    bxm = 0.5*(bx0 + bx1)
    bx = bats_interp2.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Interpolate at mid-point between current point and next point along y axis (GSE)
    x1 = batsdata.DataArray[ x_,i,j+1,k,n ]
    y1 = batsdata.DataArray[ y_,i,j+1,k,n ]
    z1 = batsdata.DataArray[ z_,i,j+1,k,n ]
    print( 'Test at y mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = batsdata.DataArray[ batsdata.varidx['bx'], i,j+1,k,n ]
    bxm = 0.5*(bx0 + bx1)
    bx = bats_interp2.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Interpolate at mid-point between current point and next point along z axis (GSE)
    x1 = batsdata.DataArray[ x_,i,j,k+1,n ]
    y1 = batsdata.DataArray[ y_,i,j,k+1,n ]
    z1 = batsdata.DataArray[ z_,i,j,k+1,n ]
    print( 'Test at z mid-pt: ', x1,y1,z1 )
    
    xm = 0.5*(x0+x1)
    ym = 0.5*(y0+y1)
    zm = 0.5*(z0+z1)
    bx1 = batsdata.DataArray[ batsdata.varidx['bx'], i,j,k+1,n ]
    bxm = 0.5*(bx0 + bx1)
    bx = bats_interp2.interpolator( (xm,ym,zm), 'bx')[0]
    print( 'bx: ', bxm, 'bx diff: ', bxm-bx )

    # Compare results to other BATSRUS interpolator        
    
    NUM = 10000
    bx1 = np.zeros(NUM)
    bx2 = np.zeros(NUM)
    
    for i in range(NUM):
        i1 = randint(0,nI-1)
        j1 = randint(0,nJ-1)
        k1 = randint(0,nK-1)
        n1 = randint(0,nBlock-1)
        i2 = randint(0,nI-1)
        j2 = randint(0,nJ-1)
        k2 = randint(0,nK-1)
        n2 = randint(0,nBlock-1)

        x1 = batsdata.DataArray[ x_,i1,j1,k1,n1 ]
        y1 = batsdata.DataArray[ y_,i1,j1,k1,n1 ]
        z1 = batsdata.DataArray[ z_,i1,j1,k1,n1 ]
        x2 = batsdata.DataArray[ x_,i2,j2,k2,n2 ]
        y2 = batsdata.DataArray[ y_,i2,j2,k2,n2 ]
        z2 = batsdata.DataArray[ z_,i2,j2,k2,n2 ]

        x0 = 0.5*(x1+x2)
        y0 = 0.5*(y1+y2)
        z0 = 0.5*(z1+z2)
        
        r = np.sqrt( x0**2 + y0**2 + z0**2 )
        
        if r > batsdata.rCurrents:
            bx2[i] = bats_interp2.interpolator( (x0,y0,z0), 'bx')[0]
            bx1[i] = bats_interp.interpolator( (x0,y0,z0), 'bx')[0]
            
    import matplotlib.pyplot as plt

    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Linear bx')
    plt.ylabel('Kamono bx')
    plt.show()

    plt.plot( bx1, bx2, 'r+')
    plt.xlabel('Linear bx')
    plt.ylabel('Kamono bx')
    plt.xlim((-50,50))
    plt.ylim((-50,50))
    plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 07:21:06 2024

@author: Dean Thomas
"""

import logging
import numpy as np
from copy import deepcopy

from deltaB.OCTREE_BLOCK_GRID._interpolate_amrdata import ffi
from deltaB.OCTREE_BLOCK_GRID._interpolate_amrdata import lib

class BATSRUS_interpolator():
    """Class to interpolate BATSRUS results.  Heavily-based on Kamodo code
    written by Rebecca Ringuette.  Based on code in Kamodo swmfgm_4D.py and
    the associated OCTREE_BLOCK_GRID library.
    '''

    """
    def __init__(self, batsrus):
        """Initialize batsrus_interpolator class
            
        Inputs:
            batsrus = class swmfio reading of BATSRUS file, contains SWMF results
                 
        Outputs:
            None
        """
        # logging.info('Initializing batsrus interpolator class') 

        # Store instance data
        self.batsrus = batsrus
        self.octree = {}
        self.var_data = {}
        
        self.var_dict = dict(self.batsrus.varidx)

        self.x = np.array(self.batsrus.data_arr[:, self.var_dict['x']][:])
        self.y = np.array(self.batsrus.data_arr[:, self.var_dict['y']][:])
        self.z = np.array(self.batsrus.data_arr[:, self.var_dict['z']][:])

        return

    def setup_octree(self, xx, yy, zz):
        '''
        This function requires _interpolate_amrdata*.so in
        readers/OCTREE_BLOCK_GRID/
        '''
        NX = self.batsrus.nI
        NY = self.batsrus.nJ
        NZ = self.batsrus.nK

        Ncell = int(len(xx))

        N_blks = int(len(xx)/(NX*NY*NZ))

        # initialize cffi arrays with Python lists may be slightly faster
        self.octree = {}
        x = ffi.new("float[]", list(xx))
        y = ffi.new("float[]", list(yy))
        z = ffi.new("float[]", list(zz))
        # 2 elements per block, smallest blocks have 2x2x2=8 positions
        # at cell corners (GUMICS), so maximum array size is N/4
        # BATSRUS blocks have at least 4x4x4 cell centers
        self.octree['xr_blk'] = ffi.new("float[]", 2*N_blks)
        self.octree['yr_blk'] = ffi.new("float[]", 2*N_blks)
        self.octree['zr_blk'] = ffi.new("float[]", 2*N_blks)
        self.octree['x_blk'] = ffi.new("float[]", NX*N_blks)
        self.octree['y_blk'] = ffi.new("float[]", NY*N_blks)
        self.octree['z_blk'] = ffi.new("float[]", NZ*N_blks)
        self.octree['box_range'] = ffi.new("float[]", 7)
        self.octree['NX'] = ffi.new("int[]", [NX])
        self.octree['NY'] = ffi.new("int[]", [NY])
        self.octree['NZ'] = ffi.new("int[]", [NZ])

        N_blks = lib.xyz_ranges(Ncell, x, y, z, self.octree['xr_blk'],
                                self.octree['yr_blk'], self.octree['zr_blk'],
                                self.octree['x_blk'], self.octree['y_blk'],
                                self.octree['z_blk'], self.octree['box_range'],
                                self.octree['NX'], self.octree['NY'],
                                self.octree['NZ'], 1)

        if N_blks < 0:
            print('NX:', list(self.octree['NX']))
            print('NY:', list(self.octree['NY']))
            print('NZ:', list(self.octree['NZ']))
            print('Y:', list(y[0:16]))
            print('Z:', list(z[0:16]))
            print('X_blk:', list(self.octree['x_blk'][0:16]))
            print('Y_blk:', list(self.octree['y_blk'][0:16]))
            print('Z_blk:', list(self.octree['z_blk'][0:16]))
            raise IOError("Block shape and size was not determined.")

        self.octree['N_blks'] = ffi.new("int[]", [N_blks])

        N_octree = int(N_blks*8/7)
        self.octree['octree_blocklist'] = ffi.new("octree_block[]", N_octree)

        dx_blk = np.zeros(N_blks, dtype=float)
        dy_blk = np.zeros(N_blks, dtype=float)
        dz_blk = np.zeros(N_blks, dtype=float)

        for i in range(0, N_blks):
            dx_blk[i] = self.octree['xr_blk'][2*i+1]-self.octree['xr_blk'][2*i]
            dy_blk[i] = self.octree['yr_blk'][2*i+1]-self.octree['yr_blk'][2*i]
            dz_blk[i] = self.octree['zr_blk'][2*i+1]-self.octree['zr_blk'][2*i]

        dxmin_blk = min(dx_blk)
        dxmax_blk = max(dx_blk)
        dymin_blk = min(dy_blk)
        dymax_blk = max(dy_blk)
        dzmin_blk = min(dz_blk)
        dzmax_blk = max(dz_blk)
        XMIN = self.octree['box_range'][0]
        XMAX = self.octree['box_range'][1]
        YMIN = self.octree['box_range'][2]
        YMAX = self.octree['box_range'][3]
        ZMIN = self.octree['box_range'][4]
        ZMAX = self.octree['box_range'][5]
        p1 = int(np.floor((XMAX-XMIN)/dxmax_blk+0.5))
        p2 = int(np.floor((YMAX-YMIN)/dymax_blk+0.5))
        p3 = int(np.floor((ZMAX-ZMIN)/dzmax_blk+0.5))
        while ((int(p1/2)*2 == p1)
               & (int(p2/2)*2 == p2)
               & (int(p3/2)*2 == p3)
               ):
            p1 = int(p1/2)
            p2 = int(p2/2)
            p3 = int(p3/2)

        MAX_AMRLEVEL = int(np.log((XMAX-XMIN)/(p1*dxmin_blk))/np.log(2.)+0.5)+1
        self.octree['MAX_AMRLEVEL'] = MAX_AMRLEVEL

        self.octree['numparents_at_AMRlevel'] = ffi.new("int[]",
                                                   N_blks*(MAX_AMRLEVEL+1))
        self.octree['block_at_AMRlevel'] = ffi.new("int[]",
                                              N_blks*(MAX_AMRLEVEL+1))
        success = int(-1)
        success = lib.setup_octree(N_blks,
                                   self.octree['xr_blk'], self.octree['yr_blk'],
                                   self.octree['zr_blk'], MAX_AMRLEVEL,
                                   self.octree['box_range'],
                                   self.octree['octree_blocklist'], N_octree,
                                   self.octree['numparents_at_AMRlevel'],
                                   self.octree['block_at_AMRlevel'])

        return (self.octree)

    def register_variable(self, varname):
        '''Creates interpolator for the indicated dataset.'''

        # logging.info('Initializing batsrus interpolator variable') 
             
        # initialize octree object
        if not self.octree:
            self.octree = self.setup_octree(self.x, self.y, self.z)

            # update which octree arrays the library points to
            lib.setup_octree_pointers(self.octree['MAX_AMRLEVEL'],
                                      self.octree['octree_blocklist'],
                                      self.octree['numparents_at_AMRlevel'],
                                      self.octree['block_at_AMRlevel'])

        # store varname data to be interpolated in dictionary
        self.var_data[varname] = ffi.new("float[]", 
                    list(self.batsrus.data_arr[:, self.var_dict[varname]][:]))

        return

    # assign custom interpolator: Lutz Rastaetter 2021
    def interp(self, xvec, varname):
        if not isinstance(xvec, np.ndarray):
            xvec = np.array(xvec)

        X, Y, Z = xvec.T  # xvec can be used like this
        if not isinstance(X, np.ndarray):
            X = np.array([X])
            Y = np.array([Y])
            Z = np.array([Z])

        xpos = ffi.new("float[]", list(X))
        ypos = ffi.new("float[]", list(Y))
        zpos = ffi.new("float[]", list(Z))
        npos = len(X)
        return_data = list(np.zeros(npos, dtype=float))
        return_data_ffi = ffi.new("float[]", return_data)

        IS_ERROR = lib.interpolate_amrdata_multipos(
            xpos, ypos, zpos, npos, self.var_data[varname], return_data_ffi)
        if IS_ERROR != 0:
            print("Warning: SWMF/BATSRUS interpolation failed.")

        return_data[:] = list(return_data_ffi)

        return return_data


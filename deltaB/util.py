#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:21:09 2023

@author: Dean Thomas
"""

import numpy as np
from os.path import exists
from os import makedirs
import os.path
import logging

def get_spherical_components(v_cart, x_cart):
    x = x_cart[0]
    y = x_cart[1]
    z = x_cart[2]
    vx = v_cart[0]
    vy = v_cart[1]
    vz = v_cart[2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    L = np.sqrt(x**2 + y**2)

    v_r = (x*vx + y*vy + z*vz)/r
    v_theta = ((x*vx + y*vy)*z - (L**2)*vz)/(r*L)
    v_phi = (-y*vx + x*vy)/L

    return np.array([v_r, v_theta, v_phi])

def get_NED_components(v_cart, x_cart):
    v_sph = get_spherical_components(v_cart, x_cart)
    v_north = -v_sph[1]
    v_east  = +v_sph[2]
    v_down  = -v_sph[0]
    return v_north, v_east, v_down

##############################################################################
##############################################################################
# nez from weigel
# https://github.com/rweigel/magnetovis/blob/newapi2/magnetovis/Sources/BATSRUS_dB_demo.py#L45
#
# Modified to use spacepy coordinate transforms instead of hxform transforms
# hxform has a bug and for recent dates yields incorrect transformations
#
# Modified from north-east-zenith to north-east-down to get a right-handed
# coordinate system
##############################################################################
##############################################################################

def ned(time, pos, csys, pole_csys='GEO'):
    """Unit vectors in geographic north, east, and down directions
    
    Inputs:
        time = time in ISO format -> '2002-02-25T12:20:30'
        
        pos = position where we want the north-east-zenith unit vectors
        
        csys = coordinate system of pos, e.g., GEO, GSE, GSM, SM, MAG, ECIMOD
        
        pole_csys = coordinate system of north pole, default is 'GEO'
        
    Outputs:
        n_geo, e_geo, d_geo = north, east, down geographic unit vectors
    """
    
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    
    # Geographic z axis in csys
    Z001 = coord.Coords([[0,0,1]], pole_csys, 'car', use_irbem=False)
    Z001.ticks = Ticktock([time], 'ISO')
    Z = Z001.convert(csys, 'car')
    
    # zenith direction ("up")
    z_geo = pos/np.linalg.norm(pos)
    
    e_geo = np.cross(Z.data[0], z_geo)
    e_geo = e_geo/np.linalg.norm(e_geo)
    
    n_geo = np.cross(z_geo, e_geo)
    n_geo = n_geo/np.linalg.norm(n_geo)
    
    d_geo = - z_geo
    
    # print(f"Unit vectors for Geographic N, E, and D in {csys}:")
    # print(f"North: {n_geo}")
    # print(f"East:  {e_geo}")
    # print(f"Down:  {d_geo}")
    
    return n_geo, e_geo, d_geo

# def date_timeISO(y, m, d, hh, mm, ss):
#     """Pull date and time from file basename
#     Inputs:
#         y, m, d, hh, mm, ss = integers: year, month, , hour, minute, second
        
#     Outputs:
#         time in ISO format -> '2002-02-25T12:20:30'
#      """
 
#     timeiso = str(y) + '-' + str(m).zfill(2) + '-' + str(d).zfill(2) + 'T' + \
#         str(hh).zfill(2) +':' + str(mm).zfill(2) + ':' + str(ss).zfill(2)

#     return timeiso

def date_timeISO(time):
    """Pull date and time from file basename
    Inputs:
        time = list of integers: year, month, , hour, minute, second
        
    Outputs:
        time in ISO format -> '2002-02-25T12:20:30'
     """
 
    timeiso = str(time[0]) + '-' + str(time[1]).zfill(2) + '-' + str(time[2]).zfill(2) + 'T' + \
        str(time[3]).zfill(2) +':' + str(time[4]).zfill(2) + ':' + str(time[5]).zfill(2)

    return timeiso

def create_directory( target, folder ):
    """ If directory for output files does not exist, create it
    
    Inputs:
        target = main folder that will contain the "folder" subdirectory
        
        folder = basename of folder.  Complete path to folder is:
            target + folder
            
    Outputs:
        None 
     """
    path = os.path.join( target, folder )
    
    logging.info('Looking for directory: ' + path)
    if not exists(path):
        logging.info('Creating directory: ' + path)
        makedirs(path)
    return

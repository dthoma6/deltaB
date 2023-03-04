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
from copy import deepcopy
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
    Z001 = coord.Coords([[0,0,1]], pole_csys, 'car')
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
    m = (date % 10000) // 100
    d = date % 100

    # Convert time to a integers
    t = int(words[1])
    hh = t//10000
    mm = (t % 10000) // 100
    ss = t % 100

    # logging.info(f'Time: {date} {t} Year: {y} Month: {n} Day: {d} Hours: {h} Minutes: {m} Seconds: {s}')

    return y, m, d, hh, mm, ss

def date_timeISO(file):
    """Pull date and time from file basename

    Inputs:
        file = basename of file.
        
    Outputs:
        year, month, day, hour, minute, second in ISO format -> '2002-02-25T12:20:30'
     """
    words = file.split('-')

    date = int(words[0].split('e')[1])
    y = date//10000
    m = (date % 10000) // 100
    d = date % 100

    # Convert time to a integers
    t = int(words[1])
    hh = t//10000
    mm = (t % 10000) // 100
    ss = t % 100

    timeiso = str(y) + '-' + str(m).zfill(2) + '-' + str(d).zfill(2) + 'T' + \
        str(hh).zfill(2) +':' + str(mm).zfill(2) + ':' + str(ss).zfill(2)

    return timeiso

def get_files(orgdir, reduce=False, base='3d__*', file_type='out'):
    """Create a list of files that we will process.  Look in the basedir directory,
    and get list of file basenames.

    Inputs:
        orgdir = path to directory containing input files
        
        reduce = boolean, reduce the number of files examined.

        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out' or dirpath + base + '.cdf'
            
        file_type = look for either 'out' or 'cdf' file extensions
                    
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

    assert( file_type == 'out' or file_type == 'cdf' )
    
    # Look for .out files or .cdf files as appropriate  
    if file_type == 'out':
        l1 = glob.glob(base + '.out')
    else: 
        l1 = glob.glob(base + '.cdf')


    # Strip off extension
    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    # If we have a large number of input files, e.g., one a minute, reduce 
    # the number by accepting those only every 15 minutes
    if reduce: 
        l2 = deepcopy(l1) 
        for i in range(len(l2)):
            y,m,d,hh,mm,ss = date_time(l2[i])
            if( mm % 15 != 0 ):
                l1.remove(l2[i])

    l1.sort()

    return l1

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

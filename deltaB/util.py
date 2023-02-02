#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:21:09 2023

@author: Dean Thomas
"""

import numpy as np
from os.path import exists
from os import makedirs
from copy import deepcopy

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

def ned(time, pos, csys):
    """Unit vectors in geographic north, east, and down directions
    
    Inputs:
        time = time in ISO format -> '2002-02-25T12:20:30'
        
        pos = position where we want the north-east-zenith unit vectors
        
        csys = coordinate system of pos, e.g., GEO, GSE, GSM, SM, MAG, ECIMOD
        
    Outputs:
        n_geo, e_geo, d_geo = north, east, down geographic unit vectors
    """
    
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock
    
    # Geographic z axis in csys
    Z001 = coord.Coords([[0,0,1]], 'GEO', 'car')
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

def get_files(orgdir, reduce=False, base='3d__*'):
    """Create a list of files that we will process.  Look in the basedir directory,
    and get list of file basenames.

    Inputs:
        orgdir = path to directory containing input files
        
        reduce = boolean, reduce the number of files examined.

        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out'
                    
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

    l1 = glob.glob(base + '.out')

    # Strip off extension
    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    # Colaba incliudes 697 files, reduce the number by
    # accepting those only every 15 minutes
    if reduce: 
        l2 = deepcopy(l1) 
        for i in range(len(l2)):
            y,m,d,hh,mm,ss = date_time(l2[i])
            if( mm % 15 != 0 ):
                l1.remove(l2[i])

    l1.sort()

    return l1

def get_files_unconverted(tgtsubdir, orgdir, tgtdir, reduce=False, base='3d__*'):
    """Create a list of files that we will process.  This routine is used when
    some files have been process and others have not, e.g., the program crashed.
    Since the output files of other routines use the same basenames as the output
    files, we compare the files in the input directory (orgdir) to those in the
    output directory (tgtdir).  From this, we create a list of unprocessed files.

    Inputs:
        base = start of BATSRUS files including wildcards.  Complete path to file is:
            dirpath + base + '.out'
            
        orgdir = path to directory containing input files
        
        tgtdir = path to directory containing output files
        
        tgtsubdir = the tgtdir contains multiple subdirectories containing output
            files from various routines.  tgtdir + tgtsubdir is the name of the
            directory with the output files that we will compare
            
        reduce = boolean, reduce the number of files examined.

        base = start of BATSRUS files including wildcards.  Complete path to file is:
           dirpath + base + '.out'
                   
    Outputs:
        l = list of file basenames
    """
    import os
    import glob

    # In this routine we compare the list of .out input files and .png files
    # to determine what has already been processed.  Look at all *.out
    # files and remove from the list (l1) all of the files that already have
    # been converted to .png files.  The unremoved files are unconverted files.

    os.chdir(orgdir)
    l1 = glob.glob(base + '.out')

    # Look at the png files in directory
    if not exists(tgtdir + tgtsubdir):
        makedirs(tgtdir + tgtsubdir)
    os.chdir(tgtdir + tgtsubdir)
    l2 = glob.glob(base + '.png')

    for i in range(len(l1)):
        l1[i] = (l1[i].split('.'))[0]

    for i in range(len(l2)):
        l2[i] = (l2[i].split('.'))[0]

    for i in l2:
        l1.remove(i)

    # Colaba incliudes 697 files, reduce the number by
    # accepting those only every 15 minutes
    if reduce: 
        l3 = deepcopy(l1) 
        for i in range(len(l3)):
            y,m,d,hh,mm,ss = date_time(l3[i])
            if( mm % 15 != 0 ):
                l1.remove(l3[i])

    l1.sort()

    return l1


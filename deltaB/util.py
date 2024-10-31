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

def get_mhd_file_time(filepath):
    """From the file "*_cdf_list" read the time associated with the
    file, filepath.  *_cdf_list is in the same directory as the CDF files.
     
    Inputs:
        filepath = path to CDF file that we're processing
         
    Outputs:
        Either:
        
        Return time associated with file as a tuple: YYYY, Month, Day, Hour,
            Minute, Second
    
        Return -1 if time not found
    """
    logging.info('Obtain time associated with MHD file')
    
    dirname = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    filepathsplit = filepath.split('/')
    
    # Try multiple name variants
    cdflist = os.path.join( dirname, filepathsplit[-3] + '_GM_cdf_list')
    if not os.path.isfile(cdflist):
        cdflist = os.path.join( dirname, filepathsplit[-3] + '_IE_cdf_list')
    if not os.path.isfile(cdflist):
        cdflist = os.path.join( dirname, filepathsplit[-3] + '_cdf_list')
    elif not os.path.isfile(cdflist):
        return -1
        
    # Read cdflist and find entry associated with filepath
    file = open(cdflist)
    lines = file.readlines()

    for line in lines:
        line = line.strip()
        linea = line.split()
        if linea[0].endswith('.cdf') == False:
            continue

        datea = linea[2].split("/")
        timea = linea[4].split(":")
        time = (int(datea[0]), int(datea[1]), int(datea[2]), int(timea[0]), int(timea[1]), int(timea[2]))

        if base == linea[0]: return time
        
    return -1

def setup(info):
    """Derived from magnetopost setup.  Reads the file "*_GM_cdf_list" 
    and generates a dict of filenames and associated times, which added to info.
    *_GM_cdf_list is in the same directory as the CDF files.
     
    Inputs:
        info = python dict with information on this run
         
    Outputs:
        Added info['files'] dict
    """
    logging.info('Add python dict on MHD files to info')

    assert os.path.exists(info["dir_run"]), "dir_run = " + info["dir_run"] + " not found"
    assert info['file_type'] == 'cdf', "Setup only handles CDF files"
    # assert not info.get('rCurrents'), "info['rCurrents'] exists, and should not.  Delete from info."

    info['files'] = {}

    for subdir in ["GM_CDF", "IONO-2D_CDF"]:
        if subdir == "GM_CDF" and exists( os.path.join(info['dir_run'], subdir) ):
            # Note lower case 'cdf' in _GM_cdf_list
            file = open(os.path.join(info['dir_run'], subdir, info['run_name'] + '_GM_cdf_list'), 'r')
            key = "magnetosphere"
        if subdir == "IONO-2D_CDF" and exists( os.path.join(info['dir_run'], subdir) ):
            # Note upper case 'CDF' in _GM_CDF_list
            file = open(os.path.join(info['dir_run'], subdir, info['run_name'] + '_IE_CDF_list'), 'r')
            key = "ionosphere"

        if exists( os.path.join(info['dir_run'], subdir) ):
            info['files'][key] = {}
            lines = file.readlines()
    
            for line in lines:
                line = line.strip()
                linea = line.split()
                if linea[0].endswith('.cdf') == False:
                    continue
    
                datea = linea[2].split("/")
                timea = linea[4].split(":")
                time = (int(datea[0]), int(datea[1]), int(datea[2]), int(timea[0]), int(timea[1]), int(timea[2]))
    
                info['files'][key][time] = os.path.join(info['dir_run'], subdir, linea[0])

    return


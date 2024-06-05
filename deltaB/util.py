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

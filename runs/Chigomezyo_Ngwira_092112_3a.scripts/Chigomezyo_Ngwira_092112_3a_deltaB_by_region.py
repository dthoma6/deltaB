#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:57:20 2023

@author: Dean Thomas
"""

import os.path
import deltaB as db

#############################################################################
# Script to calculate Bn contributions from magnetospheric regions
# Regions identified in Chigomezyo_Ngwira_092112_3a_FindBoundaries.py
#############################################################################

from Chigomezyo_Ngwira_092112_3a_info import info as info
from Chigomezyo_Ngwira_092112_3a_info import setup as setup

if __name__ == "__main__":
    
    # Magnetopause and bow shock offsets along x-axis (GSM).  Used to shift the
    # positions of the magnetopause and bow shock.  In this case, we shift
    # the bow shock to include the current layer on the sunward side of the
    # bow shock
    deltamp = 0.
    deltabs = 0.5
    
    # Neutral sheet thickness.  Divide by 2 and region is +/- half the value.
    thicknessns = 6.
    
    # Radius of near earth region
    nearradius = 6.6
    
    # The times for the files that we will process, the list can be multiple times
    # for multiple files.  
    times = ((2003, 9, 2, 0, 0, 0),
              (2003, 9, 2, 1, 0, 0),
              (2003, 9, 2, 2, 0, 0),
              (2003, 9, 2, 3, 0, 0),
              (2003, 9, 2, 4, 0, 0))

    # Point at which we will calculate the magnetic field
    point = 'Colaba'

    ###################
    # magnetopause, bow shock and neutral sheet files
    ###################

    mpfiles = ('3d__var_1_e20030902-000000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-010000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-020000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-030000-000.out.cdf.30.0-91.magnetopause.pkl',
                '3d__var_1_e20030902-040000-000.out.cdf.30.0-91.magnetopause.pkl')
 
    bsfiles = ('3d__var_1_e20030902-000000-000.out.cdf.64-65.bowshock.pkl',
                '3d__var_1_e20030902-010000-000.out.cdf.64-65.bowshock.pkl',
                '3d__var_1_e20030902-020000-000.out.cdf.64-65.bowshock.pkl',
                '3d__var_1_e20030902-030000-000.out.cdf.64-65.bowshock.pkl',
                '3d__var_1_e20030902-040000-000.out.cdf.64-65.bowshock.pkl')
 
    nsfiles = ('3d__var_1_e20030902-000000-000.out.cdf.200.0-201.neutralsheet.pkl',
                '3d__var_1_e20030902-010000-000.out.cdf.200.0-201.neutralsheet.pkl',
                '3d__var_1_e20030902-020000-000.out.cdf.200.0-201.neutralsheet.pkl',
                '3d__var_1_e20030902-030000-000.out.cdf.200.0-201.neutralsheet.pkl',
                '3d__var_1_e20030902-040000-000.out.cdf.200.0-201.neutralsheet.pkl')
 
    # Get a list of BATSRUS and RIM files. info parameters define location 
    # (dir_run) and file types.  Based on info structure from magnetopost.
    # NOTE: magnetopost code fails on this, so a modified version of setup
    # is in Chigomezyo_Ngwira_092112_3a_info
    setup(info)

    db.calc_ms_b_region2D( info, deltamp, deltabs, thicknessns, 
                             nearradius, times, mpfiles, bsfiles, nsfiles, point, 
                             createVTK=True, deltahr=5.5 )
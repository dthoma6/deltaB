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
# Regions identified in CARR_Scenarios1_FindBoundaries.py
#############################################################################

from CARR_Scenarios1_info import info as info

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
    
    ###################
    # Find the bow shock and magnetosphere
    ###################

    # The times for the files that we will process, the list can be multiple times
    # for multiple files.  
    times = ((2019, 9, 2, 5, 0, 0),
              (2019, 9, 2, 6, 0, 0),
              (2019, 9, 2, 6, 30, 0),
              (2019, 9, 2, 7, 0, 0),
              (2019, 9, 2, 8, 0, 0))

    # Point at which we will calculate the magnetic field
    point = 'Colaba'

    ###################
    # magnetopause, bow shock and neutral sheet files
    ###################

    mpfiles = ('3d__var_2_e20190902-050000-008.out.30.0-91.magnetopause.pkl',
                '3d__var_2_e20190902-060000-021.out.30.0-91.magnetopause.pkl',
                '3d__var_2_e20190902-063000-000.out.20.0-61.magnetopause.pkl',
                '3d__var_2_e20190902-070000-888.out.30.0-91.magnetopause.pkl',
                '3d__var_2_e20190902-080000-497.out.30.0-91.magnetopause.pkl')
 
    bsfiles = ('3d__var_2_e20190902-050000-008.out.64.0-65.bowshock.pkl',
                '3d__var_2_e20190902-060000-021.out.64.0-65.bowshock.pkl',
                '3d__var_2_e20190902-063000-000.out.48.0-49.bowshock.pkl',
                '3d__var_2_e20190902-070000-888.out.64.0-65.bowshock.pkl',
                '3d__var_2_e20190902-080000-497.out.64.0-65.bowshock.pkl')
 
    nsfiles = ('3d__var_2_e20190902-050000-008.out.200.0-201.neutralsheet.pkl',
                '3d__var_2_e20190902-060000-021.out.200.0-201.neutralsheet.pkl',
                '3d__var_2_e20190902-063000-000.out.200.0-201.neutralsheet.pkl',
                '3d__var_2_e20190902-070000-888.out.200.0-201.neutralsheet.pkl',
                '3d__var_2_e20190902-080000-497.out.200.0-201.neutralsheet.pkl')
 
    # # Get a list of BATSRUS and RIM files, info parameters define location 
    # # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)

    db.calc_ms_b_region2D( info, deltamp, deltabs, thicknessns, 
                             nearradius, times, mpfiles, bsfiles, nsfiles, point )

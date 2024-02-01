#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:49:17 2023

@author: Dean Thomas
"""

import os.path
from deltaB import loop_heatmapworld_ms, plot_heatmapworld_ms, \
    loop_heatmapworld_iono, plot_heatmapworld_iono, \
    loop_heatmapworld_gap, plot_heatmapworld_gap, \
    plot_heatmapworld_ms_total, \
    loop_heatmapworld_ms_by_region, plot_heatmapworld_ms_by_region, \
    plot_heatmapworld_ms_by_region_grid, plot_heatmapworld_ms_by_currents_grid, \
    plot_heatmapworld_ms_by_currents_grid2, \
    plot_histogram_ms_by_region_grid, plot_histogram_ms_by_currents_grid, \
    date_timeISO, create_directory

from CARR_Scenarios1_info import info as info

############################################################################
#
# Script to generate heatmaps for paper
#
############################################################################

if __name__ == "__main__":
    
    # Max/min of scale used in heatmaps
    VMIN = -2000
    VMAX = +2000

    # We will plot the magnitude of the B field in a lat/long grid
    # Define the grid size
    NLAT = 30
    NLONG = 60
    
    BINWIDTH = 50

    # The times for the files that we will process
    TIMES = ( (2019, 9, 2, 5, 0, 0),
              (2019, 9, 2, 6, 0, 0),
              (2019, 9, 2, 6, 30, 0),
              (2019, 9, 2, 7, 0, 0),
              (2019, 9, 2, 8, 0, 0))
 
    # Magnetopause and bow shock offsets along x-axis (GSM).  In this case,
    # used to include current layer on surface of bow shock
    deltamp = 0.
    deltabs = 0.5
    
    # Neutral sheet thickness
    thicknessns = 6.
    
    # Radius of near earth region
    nearradius = 6.6
    
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
 
    # Get a list of BATSRUS and RIM files, info parameters define location 
    # (dir_run) and file types.  See definition of info = {...} above.
    from magnetopost import util as util
    util.setup(info)
    
    # Calculate the delta B sums to get Bn contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere over a lat-long grid
    loop_heatmapworld_ms( info, TIMES, NLAT, NLONG )
    loop_heatmapworld_iono( info, TIMES, NLAT, NLONG )
    loop_heatmapworld_gap( info, TIMES, NLAT, NLONG, nR=100, useRIM=True )
    loop_heatmapworld_ms_by_region( info, TIMES, NLAT, NLONG, deltamp, deltabs, 
                            thicknessns, nearradius, 
                            mpfiles, bsfiles, nsfiles )

    # Create heatmaps plots of Bn over earth
    # plot_heatmapworld_ms( info, TIMES, VMIN, VMAX, NLAT, NLONG )
    # plot_heatmapworld_ms_total( info, TIMES, VMIN, VMAX, NLAT, NLONG, csys='GEO', threesixty=False )
    # plot_heatmapworld_iono( info, TIMES, VMIN, VMAX, NLAT, NLONG, csys='SM', threesixty=True )
    # plot_heatmapworld_gap( info, TIMES, VMIN, VMAX, NLAT, NLONG, csys='GEO', threesixty=False )
    # plot_heatmapworld_ms_by_region( info, TIMES, VMIN, VMAX, NLAT, NLONG, deltamp, deltabs, 
    #                         thicknessns, nearradius )
    
    plot_heatmapworld_ms_by_region_grid(info, TIMES, VMIN, VMAX, NLAT, NLONG, 
      deltamp, deltabs, thicknessns, nearradius)
    plot_heatmapworld_ms_by_currents_grid(info, TIMES, VMIN, VMAX, NLAT, NLONG) 
    plot_heatmapworld_ms_by_currents_grid2(info, TIMES, VMIN, VMAX, NLAT, NLONG, 
                                           axisticks=False) 
    
    # plot_histogram_ms_by_region_grid(info, TIMES, VMIN, VMAX, BINWIDTH, 
    #     deltamp, deltabs, thicknessns, nearradius)
    # plot_histogram_ms_by_currents_grid(info, TIMES, VMIN, VMAX, BINWIDTH) 
    # plot_histogram_ms_by_region_grid(info, TIMES, None, None, BINWIDTH, 
    #     deltamp, deltabs, thicknessns, nearradius, sharex=False, sharey=False)
    # plot_histogram_ms_by_currents_grid(info, TIMES, None, None, BINWIDTH, 
    #     sharex=False, sharey=False) 
    


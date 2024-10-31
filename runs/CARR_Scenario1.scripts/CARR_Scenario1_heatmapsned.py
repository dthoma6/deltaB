#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:15:52 2024

@author: Dean Thomas
"""

from deltaB import setup, loop_heatmapworldned_ms, loop_heatmapworldned_divB, \
    loop_heatmapworldned_inner, loop_heatmapworldned_outer, \
    plot_heatmapworld_helmholtz_grid

from CARR_Scenarios1_info import info as info 

# True if we compute data for heatmaps, False if we only plot the results
COMPUTE = False

############################################################################
#
# Script to generate heatmaps for paper
#
############################################################################

if __name__ == "__main__":
    
    # Max/min of scale used in heatmaps
    VMIN = -1200
    VMAX = +1200

    # We will plot the magnitude of the B field in a lat/long grid
    # Define the grid size
    NLAT = 30
    NLONG = 60
    
    # BINWIDTH = 50

    DELTAHR = None
    
    # The times for the files that we will process
    TIMES = ( (2019, 9, 2, 5, 0, 0),
              (2019, 9, 2, 6, 0, 0),
              (2019, 9, 2, 6, 30, 0),
              (2019, 9, 2, 7, 0, 0),
              (2019, 9, 2, 8, 0, 0))
     
    # Get a list of BATSRUS files. info parameters define location 
    # (dir_run) and file types. 
    setup(info)
    
    # Calculate the delta B sums to get Bn contributions from 
    # various current systems in the magnetosphere, gap region, and 
    # the ionosphere over a lat-long grid
    if COMPUTE:
        loop_heatmapworldned_ms( info, TIMES, NLAT, NLONG, deltahr=DELTAHR, maxcores=20 )
        loop_heatmapworldned_divB( info, TIMES, NLAT, NLONG, deltahr=DELTAHR, maxcores=20 )
        loop_heatmapworldned_inner( info, TIMES, NLAT, NLONG, deltahr=DELTAHR, maxcores=20 )
        loop_heatmapworldned_outer( info, TIMES, NLAT, NLONG, deltahr=DELTAHR, maxcores=20 )

    # Create heatmaps plots of Bn over earth
    plot_heatmapworld_helmholtz_grid( info, TIMES, VMIN, VMAX, NLAT, NLONG, component=r'$B_n$' )
    plot_heatmapworld_helmholtz_grid( info, TIMES, VMIN, VMAX, NLAT, NLONG, component=r'$B_e$' )
    plot_heatmapworld_helmholtz_grid( info, TIMES, VMIN, VMAX, NLAT, NLONG, component=r'$B_d$' )
 
   


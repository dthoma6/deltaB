#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 07:56:25 2022

@author: Dean Thomas
"""

import logging
logging.basicConfig(
    format='%(filename)s:%(funcName)s(): %(message)s',
    level=logging.INFO,
    datefmt='%S')

from .plotting import plotargs, plotargs_multiy, \
    plot_NxM, plot_NxM_multiy, pointcloud, sqwireframe

from .BATSRUS_dataframe import convert_BATSRUS_to_dataframe, \
    create_deltaB_spherical_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_deltaB_rCurrents_spherical_dataframe, \
    create_cumulative_sum_dataframe, \
    create_cumulative_sum_spherical_dataframe, \
    create_jrtp_cdf_dataframes, \
    create_jpp_cdf_dataframes

from .util import get_spherical_components, get_NED_components, ned, \
    date_timeISO, create_directory

from .process_ms import calc_ms_b, calc_ms_b_paraperp, loop_ms_b
from .process_gap import calc_gap_b, loop_gap_b
from .process_iono import calc_iono_b, loop_iono_b

from .coordinates import get_transform_matrix, iso2ints, GSMtoSM, SMtoGSM

from .deltaB_by_region import write_extended_vtk, find_regions, \
    calc_ms_b_region_to_files, calc_ms_b_region

from .plots2D_Bn import loop_2D_ms, loop_2D_ms_point, plot_2D_ms, \
    loop_2D_gap_iono, loop_2D_gap_iono_point, plot_2D_gap_iono, \
    plot_2D_ms_gap_iono

from .plotsHeatmapWorld_Bn import loop_heatmapworld_ms, plot_heatmapworld_ms, \
    loop_heatmapworld_ms_by_region, plot_heatmapworld_ms_by_region, \
    plot_heatmapworld_ms_total, \
    loop_heatmapworld_iono, plot_heatmapworld_iono, \
    loop_heatmapworld_gap, plot_heatmapworld_gap, \
    plot_heatmapworld_ms_by_region_grid, plot_heatmapworld_ms_by_currents_grid, \
    plot_histogram_ms_by_region_grid, plot_histogram_ms_by_currents_grid
    
from .plots2D_BATSRUS import loop_2D_BATSRUS, \
    loop_2D_BATSRUS_with_cuts, \
    loop_2D_BATSRUS_3d_cut_vtk, \
    loop_2D_BATSRUS_3d_cut_plots, \
    process_BATSRUS, \
    process_BATSRUS_with_cuts, \
    process_BATSRUS_3d_cut_plots
    
from .plots2D_Bned import plot_Bned_ms_gap_iono

from .plots2D_BATSRUSparams import loop_2D_BATSRUSparams

from .find_boundaries import findboundary_mp, findboundary_bs, findboundary_ns

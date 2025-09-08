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

from .ms_dataframe import  convert_mhd_to_dataframe, \
    create_deltaB_spherical_dataframe, \
    create_deltaB_biotsavart_dataframe, \
    create_deltaB_biotsavart_spherical_dataframe, \
    create_cumulative_sum_dataframe, \
    create_cumulative_sum_spherical_dataframe, \
    create_jrtp_cdf_dataframes, \
    create_jpp_cdf_dataframes

from .util import date_timeISO, create_directory, get_mhd_file_time, setup, \
    gunzip_to_temp

from .process_ms import calc_ms_b, calc_ms_b_paraperp, loop_ms_b
from .process_gap import calc_gap_b_sub, calc_gap_b, loop_gap_b, \
    calc_gap_b_rim_sub, calc_gap_b_rim
from .process_iono import calc_iono_b, loop_iono_b
from .process_ms_surfint_rCurrents import calc_ms_surfint_rCurrents_b, \
    loop_ms_surfint_rCurrents_b
from .process_ms_surfint_outer import calc_ms_surfint_outer_b, \
    loop_ms_surfint_outer_b
from .process_ms_divBint import calc_ms_divBint_b, loop_ms_divBint_b

from .coordinates import get_transform_matrix, iso2ints, GSMtoSM, SMtoGSM, \
    transform, get_spherical_components, get_NED_components

from .deltaB_by_region import write_extended_vtk, find_regions, \
    calc_ms_b_region2D, calc_ms_b_region

from .plots2D_Bn import loop_2D_ms, loop_2D_ms_point, plot_2D_ms, \
    loop_2D_gap_iono, loop_2D_gap_iono_point, plot_2D_gap_iono, \
    plot_2D_ms_gap_iono

from .plotsHeatmapWorld_Bn import loop_heatmapworld_ms, plot_heatmapworld_ms, \
    loop_heatmapworld_ms_by_region, plot_heatmapworld_ms_by_region, \
    plot_heatmapworld_ms_total, \
    loop_heatmapworld_iono, plot_heatmapworld_iono, \
    loop_heatmapworld_gap, plot_heatmapworld_gap, \
    plot_heatmapworld_ms_by_region_grid, plot_heatmapworld_ms_by_currents_grid, \
    plot_heatmapworld_ms_by_currents_grid2, \
    plot_histogram_ms_by_region_grid, plot_histogram_ms_by_currents_grid, \
    earth_currents_heatmap, earth_region_heatmap, \
    loop_heatmapworld_divB
    
from .plotsHeatmapWorld_Bned import loop_heatmapworldned_ms, loop_heatmapworldned_divB, \
    loop_heatmapworldned_inner, loop_heatmapworldned_outer, plot_heatmapworld_helmholtz_grid

from .plots2D_BATSRUS import loop_2D_BATSRUS, \
    loop_2D_BATSRUS_with_cuts, \
    loop_2D_BATSRUS_3d_cut_vtk, \
    loop_2D_BATSRUS_3d_cut_plots, \
    process_BATSRUS, \
    process_BATSRUS_with_cuts, \
    process_BATSRUS_3d_cut_plots
    
from .plots2D_Bned import plot_Bned_ms_gap_iono, plot_Bn_ms_gap_iono

from .plots2D_BATSRUSparams import loop_2D_BATSRUSparams

from .find_boundaries import findboundary_mp, findboundary_bs, findboundary_ns

from .magnetometers import specified_magnetometers

from .BATSRUS_data import BATSRUSdata
from .BATSRUS_dataframe import get_batsrus_data_from_cdf
from .BATSRUS_interpolator import BATSRUS_interpolator
from .BATSRUS_interpolator2 import BATSRUS_interpolator2
from .BATSRUS_to_VTK import BATSRUS_to_VTK
from .BATSRUS_surfint_outer_b import BATSRUS_surfint_outer_b
from .BATSRUS_surfint_rCurrents_b import BATSRUS_surfint_rCurrents_b
from .BATSRUS_divBint_b import BATSRUS_divBint_b
from .BATSRUS_curlB import BATSRUS_curlBtoJ

from .OpenGGCM_data import OpenGGCMdata
from .OpenGGCM_dataframe import get_openggcm_data_from_cdf
from .OpenGGCM_interpolator import OpenGGCM_interpolator
from .OpenGGCM_interpolator2 import OpenGGCM_interpolator2
from .OpenGGCM_to_VTK import OpenGGCM_to_VTK
from .OpenGGCM_surfint_outer_b import OpenGGCM_surfint_outer_b
from .OpenGGCM_surfint_rCurrents_b import OpenGGCM_surfint_rCurrents_b
from .OpenGGCM_divBint_b import OpenGGCM_divBint_b
from .OpenGGCM_curlB import OpenGGCM_curlBtoJ

from .LFM_data import LFMdata
from .LFM_dataframe import get_lfm_data_from_cdf
# from .LFM_dataframe import get_lfm_data_from_hdf
from .LFM_interpolator import LFM_interpolator
from .LFM_interpolator2 import LFM_interpolator2
from .LFM_to_VTK import LFM_to_VTK
from .LFM_curl import lfm_curl
from .LFM_surfint_outer_b import LFM_surfint_outer_b
from .LFM_surfint_rCurrents_b import LFM_surfint_rCurrents_b
from .LFM_divBint_b import LFM_divBint_b

from .MHD_to_VTK import MHD_to_VTK

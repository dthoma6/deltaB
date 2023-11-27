#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:43:30 2023

@author: Dean Thomas
"""

"""
This script calculates the delta B contributions from each point in the BATSRUS
grid.  The contributions are divided into regions - contributions from the
magnetopause currents, neutral sheet currents, near earth currents (e.g., ring 
currents), and currents within the region that find_boundaries examined, and
in all other points within the BATSRUS grid.
"""

from numba import jit
import os.path
import pandas as pd
import numpy as np
import swmfio
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from deltaB import convert_BATSRUS_to_dataframe, \
    create_deltaB_rCurrents_dataframe, \
    create_deltaB_spherical_dataframe, \
    create_deltaB_rCurrents_spherical_dataframe, \
    create_cumulative_sum_dataframe, \
    SMtoGSM, GSMtoSM, get_NED_components, date_timeISO, \
    create_directory

from datetime import datetime
from copy import deepcopy
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from datetime import timedelta

# info = {...} example is below

# data_dir = '/Users/dean/Documents/GitHub/deltaB/runs'

# info = {
#         "model": "SWMF",
#         "run_name": "divB_simple1",
#         "rCurrents": 3.0,
#         "rIonosphere": 1.01725,
#         "file_type": "out",
#         "dir_run": os.path.join(data_dir, "divB_simple1"),
#         "dir_plots": os.path.join(data_dir, "divB_simple1.plots"),
#         "dir_derived": os.path.join(data_dir, "divB_simple1.derived"),
#         "dir_magnetosphere": os.path.join(data_dir, "divB_simple1", "GM"),
#         "dir_ionosphere": os.path.join(data_dir, "divB_simple1", "IE")
# }


def write_extended_vtk(file_or_class, variables="all", epsilon=None, blocks=None, use_ascii=False):
    """Code borrowed from swmfio.  Some modification to incorporate region data into
    vtk file
    """
    #DT change to original code, these are already imported.
    # import numpy as np
    # import swmfio

    if isinstance(file_or_class, str):
        batsclass = swmfio.read_batsrus(file_or_class)
        fileout = file_or_class
    else:
        fileout = file_or_class.file
        batsclass = file_or_class

    variables_all = list(dict(batsclass.varidx).keys())
    # This function only has an option of writing full vector elements. So any
    # variable ending with x, y, or z is renamed to not have that ending.
    # TODO: Also check if requested variable is 'x', 'y', or 'z', which is always written
    # and so requesting it does not make sense.
    for v in range(len(variables_all)):
        if len(variables_all[v]) > 1 and (variables_all[v].endswith('x') or variables_all[v].endswith('y') or variables_all[v].endswith('z')):
            variables_all[v] = variables_all[v][0:-1]
    variables_all = list(set(variables_all))

    save_all = True
    if isinstance(variables, str):
        if variables == "all":
            variables = variables_all
        else:
            save_all = False
            variables = [variables]

    for variable in variables:
        assert variable in variables_all, f"'{variable}' is not in list of available variables: {variables_all}"

    swmfio.logger.info("Creating VTK data structure.")

    DA = batsclass.DataArray
    vidx = batsclass.varidx

    nI = batsclass.nI
    nJ = batsclass.nJ
    nK = batsclass.nK
    nBlock = batsclass.block2node.size
    nNode = batsclass.node2block.size

    swmfio.logger.info(" (nI, nJ, nK) = ({0:d}, {1:d}, {2:d})".format(nI, nJ, nK))
    swmfio.logger.info(" nBlock = {0:d}".format(nBlock))
    swmfio.logger.info(" nNode  = {0:d}".format(nNode))

    nVar = len(batsclass.varidx)

    assert(DA.shape == (nVar, nI, nJ, nK, nBlock))
    assert(np.isfortran(DA))

    x_blk = DA[vidx['x'],:,:,:,:]
    y_blk = DA[vidx['y'],:,:,:,:]
    z_blk = DA[vidx['z'],:,:,:,:]
    
    is_selected = np.full(nBlock, True, dtype=bool)

    if epsilon is not None:
        is_selected = epsilon == x_blk[1,0,0, :] - x_blk[0,0,0, :]

    if blocks is not None:
        blocks = np.array(blocks)
        is_selected[:] = False
        is_selected[blocks] = True

    cell_data = []
    for vv in ['b','j','u','b1']:
        if not vv in variables: continue
        cell_data.append(
            {
                "name" : vv,
                "texture" : "VECTORS",
                "array" : np.column_stack([DA[vidx[vv+'x'],:,:,:,is_selected].ravel(),
                                           DA[vidx[vv+'y'],:,:,:,is_selected].ravel(),
                                           DA[vidx[vv+'z'],:,:,:,is_selected].ravel()])
            })
    #DT change from original code
    #DT   for sv in ['rho','p', 'measure']:
    for sv in ['rho','p', 'measure', 'boundary', 'magnetosheath', 'neutralsheet', 
               'nearearth', 'region', 'bowshock', 'magnetopause', 'jphi']:
        if not sv in variables: continue
        cell_data.append(
            {
                "name" : sv,
                "texture" : "SCALARS",
                "array" : DA[vidx[sv],:,:,:,is_selected].ravel()
            })

    nSelected = np.count_nonzero(is_selected)

    swmfio.logger.info(" nSelected = {0:d} (of {1:d})".format(nSelected, nBlock))

    block_id = np.full(nSelected*(nI)*(nJ)*(nK), -1, dtype=np.int32)
    all_vertices = np.full((nSelected*(nI+1)*(nJ+1)*(nK+1), 3), np.nan, dtype=np.float32)
    startOfBlock = 0    # Counter of points in block
    cellIndexStart = 0  # Counter of cells in block
    swmfio.logger.info(" Creating block grids.")
    for iBlockP in range(nBlock):

        swmfio.logger.debug(f"  Creating grid for block #{iBlockP+1}/{nBlock+1}")

        if blocks is not None and iBlockP > blocks[-1]:
            swmfio.logger.debug("  iBlockP > blocks[-1]. Done.")
            break

        if not is_selected[iBlockP]:
            swmfio.logger.debug(f"  Block #{iBlockP+1} not selected. Omitting.")
            continue

        block_id[cellIndexStart:cellIndexStart+nI*nJ*nK] = iBlockP
        cellIndexStart = cellIndexStart + nI*nJ*nK

        gridspacing = x_blk[1,0,0, iBlockP] - x_blk[0,0,0, iBlockP]

        xmin = x_blk[0,0,0, iBlockP] - gridspacing/2.
        ymin = y_blk[0,0,0, iBlockP] - gridspacing/2.
        zmin = z_blk[0,0,0, iBlockP] - gridspacing/2.

        xmax = x_blk[nI-1,0   ,0   , iBlockP] + gridspacing/2.
        ymax = y_blk[0   ,nJ-1,0   , iBlockP] + gridspacing/2.
        zmax = z_blk[0   ,0   ,nK-1, iBlockP] + gridspacing/2.

        swmfio.logger.debug("    (x, y, z) min = ({0:.1f}, {1:.1f}, {2:.1f})".format(xmin, ymin, zmin))
        swmfio.logger.debug("    (x, y, z) max = ({0:.1f}, {1:.1f}, {2:.1f})".format(xmax, ymax, zmax))

        grid = np.mgrid[float(xmin):float(xmax+gridspacing):float(gridspacing),
                        float(ymin):float(ymax+gridspacing):float(gridspacing),
                        float(zmin):float(zmax+gridspacing):float(gridspacing) ]
        grid = np.array(grid.reshape((3,(nI+1)*(nJ+1)*(nK+1))).transpose(), order='C')

        all_vertices[startOfBlock:startOfBlock+(nI+1)*(nJ+1)*(nK+1), :] = grid
        startOfBlock += (nI+1)*(nJ+1)*(nK+1)

    swmfio.logger.info(" Created block grids.")

    swmfio.logger.info(" Checking that vertices are unique")
    unique_vertices, pointTo = np.unique(all_vertices, axis=0, return_inverse=True)
    assert(np.all( unique_vertices[pointTo, :] == all_vertices ))
    swmfio.logger.info(" Checked that vertices are unique")

    swmfio.logger.info(" Creating nodes of blocks")
    # Nodes of blocks
    loc_in_block = np.arange((nI+1)*(nJ+1)*(nK+1)).reshape( ((nI+1),(nJ+1),(nK+1)) )
    swmfio.logger.info(" Created nodes of blocks")

    cells = []
    startOfBlock = 0

    celltype = 'VOXEL' # or HEXAHEDRON
    swmfio.logger.info(f' Creating {celltype}s.')
    for iBlockP in range(nBlock):

        swmfio.logger.debug(f"  Creating cells for block #{iBlockP+1}/{nBlock+1}")

        if blocks is not None and iBlockP > blocks[-1]:
            swmfio.logger.debug("  iBlockP > blocks[-1]. Done.")
            break

        if not is_selected[iBlockP]:
            swmfio.logger.debug(f"  Block #{iBlockP+1} not selected. Omitting.")
            continue

        # TODO: These loops can be vectorized; or, use njit.
        for i in range(nI):
            for j in range(nJ):
                for k in range(nK):

                    if celltype == 'VOXEL':
                        cells.append(
                             (pointTo[startOfBlock+loc_in_block[i  ,j  ,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j  ,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j+1,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j+1,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j  ,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j  ,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j+1,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j+1,k+1]] )
                            )
                    if celltype == 'HEXAHEDRON':
                        cells.append(
                             (pointTo[startOfBlock+loc_in_block[i  ,j  ,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j  ,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j+1,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j+1,k  ]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j  ,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j  ,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i+1,j+1,k+1]] ,
                              pointTo[startOfBlock+loc_in_block[i  ,j+1,k+1]] )
                            )

        startOfBlock += (nI+1)*(nJ+1)*(nK+1)

    cells = np.array(cells, dtype=int)

    swmfio.logger.info(f' Created {celltype}s.')    

    swmfio.logger.info("Created VTK data structure.")

    if 'block_id' in variables:
        cell_data.append(
            {
                "name" : "block_id",
                "texture" : "SCALARS",
                "array" : block_id
            })

    if use_ascii:
        ftype='ASCII'
    else:
        ftype='BINARY'

    extra = ""
    if epsilon is not None:
        extra = f"_epsilon={epsilon}"
    if variables is not None and save_all == False:
        extra = "_vars=" + ",".join(variables)
    if blocks is not None:
        if len(blocks) < 5:
            blockstrs = []
            for block in blocks:
                blockstrs.append(str(block))
            extra = "_blocks=" + ",".join(blockstrs)
        else:
            extra = f"_Nblocks={len(blocks)}"
    fileout = fileout + extra + ".vtk"

    debug = False
    if swmfio.logger.getEffectiveLevel() > 20:
        debug = True

    swmfio.logger.info("Writing " + fileout)

    from swmfio.vtk_export import vtk_export
    vtk_export(fileout, unique_vertices,
                    dataset='UNSTRUCTURED_GRID',
                    connectivity={'CELLS': {celltype: cells} },
                    cell_data=cell_data,
                    ftype=ftype,
                    debug=debug)

    swmfio.logger.info("Wrote " + fileout)

    return fileout

def find_regions( info, batsrus, deltamp, deltabs, thicknessns, nearradius, 
                       time, mpfile, bsfile, nsfile,   
                       interpType = 'nearest' ):
    
    """Goes through BATRSUS grid and determines which region each point in the
    BATSRUS grid lies in:
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth

    Inputs:
        info = tells the script where the data files are stored and where
            to save plots and calculated data, see example above
            
        batsrus = data from BATRUS file, read by swmfio.read_batsrus

        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
            
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]
        
        mpfile, bsfile, nsfile = filenames (located in os.path.join( info['dir_derived'], 
                'mp-bs-ns') that contain the magnetopause, bow shock, and neutral 
                sheet locations
            
        interpType = string, type of interpolation used below, either 'nearest' or 'linear'.
            Interpolation used with magnetopause, bow shock, and neutral sheet data
            read from mpfiles, bsfiles, and nsfiles.
        
    Outputs:
        other, magnetosheath, neutralsheet, nearearth, region, bowshock, 
            magnetopause = numpy arrays identifying the region for each point 
            in the BATSRUS grid
    """

    logging.info('Identifying regions...')

    # Extract data from BATSRUS
    var_dict = dict(batsrus.varidx)

    # Read magnetopause pkl file
    dfmp = pd.read_pickle(os.path.join( info['dir_derived'], 'mp-bs-ns', mpfile))
    
    # Replace NaN values with large neg. values to improve interpolation.
    # NaNs in magnetopause represent values where we could not find a boundary
    xmp = np.nan_to_num( dfmp['x'], nan=-10000.)
    ymp = np.nan_to_num( dfmp['y'], nan=-10000.)
    zmp = np.nan_to_num( dfmp['z'], nan=-10000.)
    
    ymp_min = np.min(ymp)
    ymp_max = np.max(ymp)
    zmp_min = np.min(zmp)
    zmp_max = np.max(zmp)

    # Set up 2D interpolation of magnetopause data
    # Note, the reordering of x,y,z to y,z,x because x in magnetopause df is
    # a function of y,z
    if interpType == 'linear':
        interpmp = LinearNDInterpolator(list(zip(ymp, zmp)), xmp )
    else:
        interpmp = NearestNDInterpolator(list(zip(ymp, zmp)), xmp )

    # Read bow shock pkl file
    dfbs = pd.read_pickle(os.path.join( info['dir_derived'], 'mp-bs-ns', bsfile))
    
    # Replace NaN values with large neg. values to improve interpolation.
    # NaNs in bow shock represent values where we could not find a boundary
    xbs = np.nan_to_num( dfbs['x'], nan=-10000.)
    ybs = np.nan_to_num( dfbs['y'], nan=-10000.)
    zbs = np.nan_to_num( dfbs['z'], nan=-10000.)
     
    ybs_min = np.min(ybs)
    ybs_max = np.max(ybs)
    zbs_min = np.min(zbs)
    zbs_max = np.max(zbs)
        
    # Set up 2D interpolation of bow shock data
    # Note, the reordering of x,y,z to y,z,x because x in bow shock df is
    # a function of y,z
    if interpType == 'linear':
        interpbs = LinearNDInterpolator(list(zip(ybs, zbs)), xbs )
    else:
        interpbs = NearestNDInterpolator(list(zip(ybs, zbs)), xbs )

    # Read neutral sheet pkl file
    dfns = pd.read_pickle(os.path.join( info['dir_derived'], 'mp-bs-ns', nsfile))
    
    # Replace NaN values with large neg. values to improve interpolation.
    # NaNs in magnetopause represent values where we could not find a boundary
    xns = np.nan_to_num( dfns['x'], nan=-10000.)
    yns = np.nan_to_num( dfns['y'], nan=-10000.)
    zns = np.nan_to_num( dfns['z'], nan=-10000.)
    
    # Set up 2D interpolation of magnetopause data
    # Note, neutral sheet uses x,y,z order
    if interpType == 'linear':
        interpns = LinearNDInterpolator(list(zip(xns, yns)), zns )
    else:
        interpns = NearestNDInterpolator(list(zip(xns, yns)), zns )

    # We will repeatedly use the x,y,z coordinates
    x = np.array( batsrus.data_arr[:, var_dict['x']][:] )
    y = np.array( batsrus.data_arr[:, var_dict['y']][:] )
    z = np.array( batsrus.data_arr[:, var_dict['z']][:] )
    r = np.sqrt( x**2 + y**2 + z**2 )
    
    # Set up memory to record whether each BATSRUS grid point is in which region.
    other = np.zeros( len(x), dtype=np.float32 ) # Within boundary but no another region
    bowshock = np.zeros( len(x), dtype=np.float32 ) # Anti-sunward of bow shock
    magnetopause = np.zeros( len(x), dtype=np.float32 ) # Anti-sunward of magnetopause
    magnetosheath = np.zeros( len(x), dtype=np.float32 ) # Between bow shock and magnetopause
    neutralsheet = np.zeros( len(x), dtype=np.float32 ) # Near the neutral sheet
    nearearth = np.zeros( len(x), dtype=np.float32 ) # Near earth
    region = np.zeros( len(x), dtype=np.float32 ) # Coded region, see below
   
    # Use interpolation to find magnetopause, bow shock, and neutral sheet.
    x_mp = interpmp( y, z )
    x_bs = interpbs( y, z )
    z_ns = interpns( x, y )
    
    # Determine which region associated with each point
    
    # Between the bow shock and magnetopause, aka within the magnetosheath
    # Make sure we stay within the limits of calculated bow shock and
    # magnetopause provided, e.g., np.abs(z) <= zbsMax
    bowshock[ (x < (x_bs + deltabs)) & (y <= ybs_max) & (y >= ybs_min) \
             & (z <= zbs_max) & (z >= zbs_min) ] = 1
    magnetopause[ (x < (x_mp + deltamp)) & (y <= ymp_max) & (y >= ymp_min) \
             & (z <= zmp_max) & (z >= zmp_min)] = 1
    magnetosheath[ (bowshock > 0.5) & (magnetopause < 0.5) ] = 1
    
    # Within the neutral sheet 
    # Outside of near earth, within thicknessns of neutral sheet,
    # and behind the magnetopause
    neutralsheet[ (r > nearradius) & (z <= z_ns + thicknessns/2) \
        & (z >= z_ns - thicknessns/2) & (x < 0) & (x <= x_mp + deltamp)] = 1
    
    # Near earth
    nearearth[ (r <= nearradius) & (x <= x_mp + deltamp)] = 1
    
    # Not in one of the other regions
    other[ (magnetosheath < 0.5) & (neutralsheet < 0.5) & (nearearth < 0.5)] = 1
    
    # Which region:
    # 0 - inside the BATSRUS grid, but not in one of the other regions
    # 1 - within the magnetosheath
    # 2 - within the neutral sheet
    # 3 - near earth
    region = magnetosheath + 2*neutralsheet + 3*nearearth
    
    return other, magnetosheath, neutralsheet, nearearth, region, bowshock, \
        magnetopause

    
def calc_ms_b_region_to_files( info, deltamp, deltabs, thicknessns, nearradius, 
                       times, mpfiles, bsfiles, nsfiles, point, createVTK = True, 
                       interpType = 'nearest', delta_hr = None, delta_min = None ):
    """Similar to calc_ms_b_region, this function saves significantly more data
    in a pickle and VTK files.
    
    Goes through BATRSUS grid and determines which region each point in the
    BATSRUS grid lies in:
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth
    
    Once the region is identified, the delta B contribution to the magnetic field
    from each region is calculated.

    Inputs:
        info = tells the script where the data files are stored and where
            to save plots and calculated data, see example above

        deltamp, deltabs = offset the x-values for the magnetopause (mp) or bow
            shock (bs).  Positive in positive GSM x coordinate.  Used to modify
            results for finite thickness of magnetopause and bow shock.
            
        thicknessns = region around neutral sheet to include.  As specified,
            neutral sheet is a 'plane.'  The neutral sheet region will extend
            thicknessns/2 above and below it.
            
        nearradius = sphere near earth in which we examine ring currents and other
            phenonmena
            
        times = the times associated with the files for which we will create
            heatmaps. The filepath is info['files']['magnetosphere'][bases[i]]
        
        mpfiles, bsfiles, nsfiles = list of filenames (located in 
                os.path.join( info['dir_derived'], 'mp-bs-ns') that contain the
                magnetopause, bow shock, and neutral sheet locations at times
            
        point = string identifying magnetometer location.  The actual location
            is pulled from a list in magnetopost.config
            
        createVTK = boolean, store region in VTK file along with original BATSRUS data
        
        interpType = string, type of interpolation used below, either 'nearest' or 'linear'.
            Interpolation used with magnetopause, bow shock, and neutral sheet data
            read from mpfiles, bsfiles, and nsfiles.
            
        delta_hr, delta_min = if None ignore, if number, shift ISO time by that 
            amount.  If value given, must be float.
        
    Outputs:
        None = results stored in pickle file and plot
    """
 
    # Interpolation must be nearest neighbor or linear
    assert interpType == 'nearest' or interpType == 'linear'
    
    # Make sure delta_hr and delta_min are float
    if delta_hr is not None:
        assert( type(delta_hr) == float )
    if delta_min is not None:
        assert( type(delta_min) == float )
    
    # Get the magnetometer location using list in magnetopost
    from magnetopost.config import defined_magnetometers
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    pointX = defined_magnetometers[point]
    XGEO = coord.Coords(pointX.coords, pointX.csys, pointX.ctype, use_irbem=False)

    # Set up memory for results
    length = len(times)
    
    B_time = [datetime(*time) for time in times]
    B_totalSM = np.zeros((length,3))
    B_otherSM = np.zeros((length,3))
    B_magnetosheathSM = np.zeros((length,3))
    B_neutralsheetSM = np.zeros((length,3))
    B_nearearthSM = np.zeros((length,3))
    Bned_total = np.zeros((length,3))
    Bned_other = np.zeros((length,3))
    Bned_magnetosheath = np.zeros((length,3))
    Bned_neutralsheet = np.zeros((length,3))
    Bned_nearearth = np.zeros((length,3))

    # Loop thru the files (aka times)
    for j in range(len(times)):
        # Get the magnetometer position in SM coordinates
        timeISO = date_timeISO( times[j] )
        if delta_hr is not None: timeISO = timeISO + timedelta(hours = delta_hr)
        if delta_min is not None: timeISO = timeISO + timedelta(minutes = delta_min)
        XGEO.ticks = Ticktock([timeISO], 'ISO')
        X = XGEO.convert( 'SM', 'car' )
        XSM = X.data[0]

        # We need the filepath for BATSRUS file
        filepath = info['files']['magnetosphere'][times[j]]
        logging.info(f'Parsing BATSRUS file... {os.path.basename(filepath)}')
        batsrus = swmfio.read_batsrus(filepath)

        nI = batsrus.nI
        nJ = batsrus.nJ
        nK = batsrus.nK
        nBlock = batsrus.block2node.size
        
        nVar = len(batsrus.varidx)
        
        assert(batsrus.DataArray.shape == (nVar, nI, nJ, nK, nBlock))

        # Identify region for each point in BATSRUS grid
        other, magnetosheath, neutralsheet, nearearth, region, bowshock, \
            magnetopause = find_regions( info, batsrus, deltamp, deltabs, 
                       thicknessns, nearradius, 
                       times[j], mpfiles[j], bsfiles[j], nsfiles[j],   
                       interpType = interpType)
                
        # Get j azimuthal, aka j phi.  Note, rCurrents is 0 since we want everything
        df = convert_BATSRUS_to_dataframe(batsrus, 0)
        df = create_deltaB_spherical_dataframe(df)
        
        logging.info('Adding regions to BATSRUS data...')

        # Add data to batsrus data so that we can use it elsewhere
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(other).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(magnetosheath).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(neutralsheet).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(nearearth).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(region).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(bowshock).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(magnetopause).T))
        batsrus.data_arr = np.hstack((batsrus.data_arr, np.atleast_2d(df['jphi']).T))
        
        DataArray = batsrus.data_arr.transpose()
        batsrus.DataArray = DataArray.reshape((len(batsrus.varidx)+8, nI, nJ, nK, nBlock), order='F')

        batsrus.varidx['other'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['magnetosheath'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['neutralsheet'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['nearearth'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['region'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['bowshock'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['magnetopause'] = np.int32(len(batsrus.varidx))
        batsrus.varidx['jphi'] = np.int32(len(batsrus.varidx))
        
        # Save to a vtk file for further analysis
        if createVTK: write_extended_vtk(batsrus)
        
        logging.info('Calculating delta B contributions...')

        # Convert XSM to GSM coordinates, which is what create_deltaB_rCurrents_dataframe needs
        XGSM = SMtoGSM(XSM, times[j], ctype_in='car', ctype_out='car')

        # Convert BATSRUS data to dataframes for each region
        df = convert_BATSRUS_to_dataframe(batsrus, info['rCurrents'], region=region)
        df = create_deltaB_rCurrents_dataframe( df, XGSM )
        df = create_deltaB_spherical_dataframe( df )
        df = create_deltaB_rCurrents_spherical_dataframe( df, XGSM )
        
        df_total = deepcopy(df)
        df_total = create_cumulative_sum_dataframe( df_total )
        B_totalGSM = np.array([df_total['dBxSum'].iloc[-1], df_total['dBySum'].iloc[-1], df_total['dBzSum'].iloc[-1]])
        B_totalSM[j,:] = GSMtoSM(B_totalGSM, times[j], ctype_in='car', ctype_out='car')
        Bned_total[j,:] = get_NED_components( B_totalSM[j,:], XSM )
        del df_total # Save memory
        
        df_other = deepcopy(df)
        df_other = df_other.drop(df_other[df_other['region'] > 0.5].index)
        df_other = create_cumulative_sum_dataframe( df_other )
        B_otherGSM = np.array([df_other['dBxSum'].iloc[-1], df_other['dBySum'].iloc[-1], df_other['dBzSum'].iloc[-1]])
        B_otherSM[j,:] = GSMtoSM(B_otherGSM, times[j], ctype_in='car', ctype_out='car')
        Bned_other[j,:] = get_NED_components( B_otherSM[j,:], XSM )
        del df_other # Save memory
        
        df_magnetosheath = deepcopy(df)
        df_magnetosheath = df_magnetosheath.drop(df_magnetosheath[df_magnetosheath['region'] < 0.5].index)
        df_magnetosheath = df_magnetosheath.drop(df_magnetosheath[df_magnetosheath['region'] > 1.5].index)
        df_magnetosheath = create_cumulative_sum_dataframe( df_magnetosheath )
        B_magnetosheathGSM = np.array([df_magnetosheath['dBxSum'].iloc[-1], df_magnetosheath['dBySum'].iloc[-1], df_magnetosheath['dBzSum'].iloc[-1]])
        B_magnetosheathSM[j,:] = GSMtoSM(B_magnetosheathGSM, times[j], ctype_in='car', ctype_out='car')
        Bned_magnetosheath[j,:] = get_NED_components( B_magnetosheathSM[j,:], XSM )
        del df_magnetosheath # Save memory
        
        df_neutralsheet = deepcopy(df)
        df_neutralsheet = df_neutralsheet.drop(df_neutralsheet[df_neutralsheet['region'] < 1.5].index)
        df_neutralsheet = df_neutralsheet.drop(df_neutralsheet[df_neutralsheet['region'] > 2.5].index)
        df_neutralsheet = create_cumulative_sum_dataframe( df_neutralsheet )
        B_neutralsheetGSM = np.array([df_neutralsheet['dBxSum'].iloc[-1], df_neutralsheet['dBySum'].iloc[-1], df_neutralsheet['dBzSum'].iloc[-1]])
        B_neutralsheetSM[j,:] = GSMtoSM(B_neutralsheetGSM, times[j], ctype_in='car', ctype_out='car')
        Bned_neutralsheet[j,:] = get_NED_components( B_neutralsheetSM[j,:], XSM )
        del df_neutralsheet # Save memory

        df_nearearth = deepcopy(df)
        df_nearearth = df_nearearth.drop(df_nearearth[df_nearearth['region'] < 2.5].index)
        df_nearearth = create_cumulative_sum_dataframe( df_nearearth )
        B_nearearthGSM = np.array([df_nearearth['dBxSum'].iloc[-1], df_nearearth['dBySum'].iloc[-1], df_nearearth['dBzSum'].iloc[-1]])
        B_nearearthSM[j,:] = GSMtoSM(B_nearearthGSM, times[j], ctype_in='car', ctype_out='car')
        Bned_nearearth[j,:] = get_NED_components( B_nearearthSM[j,:], XSM )
        del df_nearearth # Save memory
        
        # Double-check, this should be zero
        B_testGSM = B_totalGSM - B_otherGSM - B_magnetosheathGSM \
            - B_neutralsheetGSM - B_nearearthGSM
            
        logging.info(f'Bx Test GSM: {B_testGSM[0]}')
        logging.info(f'By Test GSM: {B_testGSM[1]}')
        logging.info(f'Bz Test GSM: {B_testGSM[2]}')

    # Create and save dataframe with results
    
    df_save = pd.DataFrame()
    df_save['Time'] =  B_time 

    df_save[r'$B_x$ Total'] = B_totalSM[:,0]
    df_save[r'$B_y$ Total'] = B_totalSM[:,1]
    df_save[r'$B_z$ Total'] = B_totalSM[:,2]

    df_save[r'$B_x$ Other'] = B_otherSM[:,0]
    df_save[r'$B_y$ Other'] = B_otherSM[:,1]
    df_save[r'$B_z$ Other'] = B_otherSM[:,2]

    df_save[r'$B_x$ Magnetosheath'] = B_magnetosheathSM[:,0]
    df_save[r'$B_y$ Magnetosheath'] = B_magnetosheathSM[:,1]
    df_save[r'$B_z$ Magnetosheath'] = B_magnetosheathSM[:,2]

    df_save[r'$B_x$ Neutral Sheet'] = B_neutralsheetSM[:,0]
    df_save[r'$B_y$ Neutral Sheet'] = B_neutralsheetSM[:,1]
    df_save[r'$B_z$ Neutral Sheet'] = B_neutralsheetSM[:,2]

    df_save[r'$B_x$ Near Earth'] = B_nearearthSM[:,0]
    df_save[r'$B_y$ Near Earth'] = B_nearearthSM[:,1]
    df_save[r'$B_z$ Near Earth'] = B_nearearthSM[:,2]

    df_save[r'$B_N$ Total'] = Bned_total[:,0]
    df_save[r'$B_E$ Total'] = Bned_total[:,1]
    df_save[r'$B_D$ Total'] = Bned_total[:,2]

    df_save[r'$B_N$ Other'] = Bned_other[:,0]
    df_save[r'$B_E$ Other'] = Bned_other[:,1]
    df_save[r'$B_D$ Other'] = Bned_other[:,2]

    df_save[r'$B_N$ Magnetosheath'] = Bned_magnetosheath[:,0]
    df_save[r'$B_E$ Magnetosheath'] = Bned_magnetosheath[:,1]
    df_save[r'$B_D$ Magnetosheath'] = Bned_magnetosheath[:,2]

    df_save[r'$B_N$ Neutral Sheet'] = Bned_neutralsheet[:,0]
    df_save[r'$B_E$ Neutral Sheet'] = Bned_neutralsheet[:,1]
    df_save[r'$B_D$ Neutral Sheet'] = Bned_neutralsheet[:,2]

    df_save[r'$B_N$ Near Earth'] = Bned_nearearth[:,0]
    df_save[r'$B_E$ Near Earth'] = Bned_nearearth[:,1]
    df_save[r'$B_D$ Near Earth'] = Bned_nearearth[:,2]

    # String used in title and filename
    parameters = ' [' + str(deltamp) +',' + str(deltabs) + ',' + str(thicknessns) + ',' + str(nearradius) + ']'

    create_directory(info['dir_derived'], 'mp-bs-ns/')
    df_save.to_pickle( os.path.join( info['dir_derived'], 'mp-bs-ns', 'delta B by region at ' \
                                    + point + parameters + '.pkl' ) )
        
    plt.scatter(x=df_save['Time'], y=df_save[r'$B_N$ Total'], marker='o' )
    plt.scatter(x=df_save['Time'], y=df_save[r'$B_N$ Other'], marker='s')
    plt.scatter(x=df_save['Time'], y=df_save[r'$B_N$ Magnetosheath'], marker='+')
    plt.scatter(x=df_save['Time'], y=df_save[r'$B_N$ Neutral Sheet'], marker='x')
    plt.scatter(x=df_save['Time'], y=df_save[r'$B_N$ Near Earth'], marker='*')
    
    dateformat = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_formatter(dateformat)
    
    labels = ['Total', 'Other', 'Magnetosheath', 'Neutral Sheet', 'Near Earth']
    plt.legend(labels=labels)
    
    plt.ylabel(r'$B_N$ (nT)')
    plt.xlabel(r'Time')
    plt.title(r'$B_N$ contribution from each region at ' + point + parameters)
    
    create_directory(info['dir_plots'], 'mp-bs-ns/')
    plt.savefig( os.path.join( info['dir_plots'], 'mp-bs-ns', 'Bn contribution from each region at ' \
                + point + parameters + '.png' ) )
        
    return

@jit(nopython=True)
def calc_ms_b_region_sub( Bx, By, Bz, region ):
    """ Subroutine for calc_ms_b_region to allow numba accelleration.  It 
    calculates the cummlative B field in each region:
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth
    
    Inputs:
        Bx, By, Bz = numpy arrays with magnetic field at each point
        
        region = numpy array specifying which region each point is in

    Outputs:
        Btot, Bother, Bms, Bns, Bne = cummulative sum for total (all regions),
            other, magnetosheath, neutral sheet, and near earth regions.
    """

    # Initialize memory for cumulative sums
    Btot = np.zeros(3)
    Bother = np.zeros(3)
    Bms = np.zeros(3)
    Bns = np.zeros(3)
    Bne = np.zeros(3)
    
    # Loop thru each point in the BATSRUS array, sum B field in each region
    for i in range(len(Bx)):
        Btot[0] = Btot[0] + Bx[i]
        Btot[1] = Btot[1] + By[i]
        Btot[2] = Btot[2] + Bz[i]

        match region[i]:
             case 0: 
                Bother[0] = Bother[0] + Bx[i]
                Bother[1] = Bother[1] + By[i]
                Bother[2] = Bother[2] + Bz[i]
        
             case 1: 
                Bms[0] = Bms[0] + Bx[i]
                Bms[1] = Bms[1] + By[i]
                Bms[2] = Bms[2] + Bz[i]
        
             case 2: 
                Bns[0] = Bns[0] + Bx[i]
                Bns[1] = Bns[1] + By[i]
                Bns[2] = Bns[2] + Bz[i]
        
             case 3: 
                Bne[0] = Bne[0] + Bx[i]
                Bne[1] = Bne[1] + By[i]
                Bne[2] = Bne[2] + Bz[i]
                
    return Btot, Bother, Bms, Bns, Bne

def calc_ms_b_region( XGSM, time, df ):
    """Goes through BATSRUS grid and the delta B contribution to the 
    magnetic field from each region is calculated.  The regions are:
    0 - inside the BATSRUS grid, but not in one of the other regions
    1 - within the magnetosheath
    2 - within the neutral sheet
    3 - near earth
    
    Results are in SM coordinates.  This function is similar to calc_ms_b in 
    process_ms.py, calc_ms_b does not divide the BATSRUS grid into regions.

    Inputs:
        XGSM = cartesian position where magnetic field will be measured (GSM coordinates)
                
        time = the time associated with the file for which we will create heatmaps. 

        df = BATSRUS dataframe, from call to convert_BATSRUS_to_dataframe
            
    Outputs:
        Bned_total[0], Bned_other[0], Bned_magnetosheath[0], Bned_neutralsheet[0], 
            Bned_nearearth[0] = the north components (SM coordinates) of the 
            magnetic field due to 1) all cells, 2) from cells in region 0, 
            3) ... in region 1, 4) ... in region 2, 5) ... in region 3
    """
 
    # Set up memory for results
    B_totalSM = np.zeros(3)
    B_otherSM = np.zeros(3)
    B_magnetosheathSM = np.zeros(3)
    B_neutralsheetSM = np.zeros(3)
    B_nearearthSM = np.zeros(3)
    Bned_total = np.zeros(3)
    Bned_other = np.zeros(3)
    Bned_magnetosheath = np.zeros(3)
    Bned_neutralsheet = np.zeros(3)
    Bned_nearearth = np.zeros(3)
    
    # Get XGSM in SM coordinates
    XSM = GSMtoSM(XGSM, time, ctype_in='car', ctype_out='car')

    logging.info('Calculating delta B contributions from each region...')

    # Convert BATSRUS data to dataframes for each region
    df = create_deltaB_rCurrents_dataframe( df, XGSM )
    df = create_deltaB_spherical_dataframe( df )
    df = create_deltaB_rCurrents_spherical_dataframe( df, XGSM )
    
    dBx = df['dBx'].to_numpy()
    dBy = df['dBy'].to_numpy()
    dBz = df['dBz'].to_numpy()
    region = df['region'].to_numpy()

    B_totalGSM, B_otherGSM, B_magnetosheathGSM, B_neutralsheetGSM, \
        B_nearearthGSM = calc_ms_b_region_sub( dBx, dBy, dBz, region)

    B_totalSM[:] = GSMtoSM(B_totalGSM, time, ctype_in='car', ctype_out='car')
    Bned_total[:] = get_NED_components( B_totalSM[:], XSM )

    B_otherSM[:] = GSMtoSM(B_otherGSM, time, ctype_in='car', ctype_out='car')
    Bned_other[:] = get_NED_components( B_otherSM[:], XSM )

    B_magnetosheathSM[:] = GSMtoSM(B_magnetosheathGSM, time, ctype_in='car', ctype_out='car')
    Bned_magnetosheath[:] = get_NED_components( B_magnetosheathSM[:], XSM )

    B_neutralsheetSM[:] = GSMtoSM(B_neutralsheetGSM, time, ctype_in='car', ctype_out='car')
    Bned_neutralsheet[:] = get_NED_components( B_neutralsheetSM[:], XSM )

    B_nearearthSM[:] = GSMtoSM(B_nearearthGSM, time, ctype_in='car', ctype_out='car')
    Bned_nearearth[:] = get_NED_components( B_nearearthSM[:], XSM )

    # Double-check, this should be zero
    B_testGSM = B_totalGSM - B_otherGSM - B_magnetosheathGSM \
        - B_neutralsheetGSM - B_nearearthGSM
        
    logging.info(f'Bx Test GSM: {B_testGSM[0]}')
    logging.info(f'By Test GSM: {B_testGSM[1]}')
    logging.info(f'Bz Test GSM: {B_testGSM[2]}')
    
    return Bned_total[0], Bned_other[0], Bned_magnetosheath[0], Bned_neutralsheet[0], \
        Bned_nearearth[0]
 
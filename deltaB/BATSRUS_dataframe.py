#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 08:11:30 2022

@author: Dean Thomas
"""
import cdflib.cdfread as cdfread
import numpy as np
import logging
import numba

from deltaB.util import get_mhd_file_time
from deltaB.BATSRUS_data import BATSRUSdata
from deltaB.BATSRUS_curlB import BATSRUS_curlBtoJ

USE_CURLB = False # Use curl of B to find current density True, 
                  # use BATSRUS current density False
                  
USE_FALSEB = False  # Use false B as test case

@numba.jit(nopython=True)
def get_batsrus_grid_sub( DataArray, varidx, nBlock, nI, nJ, nK):
    """ Subroutine for get_batsris_class_from_cdf that allows numba accelleration.  
    It determines the x,y,z cartesian grid for the cells
    
    Inputs:
        DataArray: numpy array in which the batsrus data is stored
        
        nBlock, nI, nJ, nK: number of points along x,y,z axes in cartesian grid
        
    Returns:
        x,y,z,measure,dx,dy,dz numpy arrays are returned
    """
    
    ncells = nBlock * (nI + 1) * (nJ + 1) * (nK+1)
    
    xcell = np.zeros(ncells)
    ycell = np.zeros(ncells)
    zcell = np.zeros(ncells)
    
    x_ = np.zeros(nI+1)
    y_ = np.zeros(nJ+1)
    z_ = np.zeros(nK+1)
    
    for b in range(nBlock):
        dx = DataArray[varidx['x'], 1,0,0, b] -  DataArray[varidx['x'], 0,0,0, b]
        dy = DataArray[varidx['y'], 0,1,0, b] -  DataArray[varidx['y'], 0,0,0, b]
        dz = DataArray[varidx['z'], 0,0,1, b] -  DataArray[varidx['z'], 0,0,0, b]
        
        x_[0:nI] = DataArray[varidx['x'], 0:nI,0,0, b] - dx/2.
        y_[0:nJ] = DataArray[varidx['y'], 0,0:nJ,0, b] - dy/2.
        z_[0:nJ] = DataArray[varidx['z'], 0,0,0:nK, b] - dz/2.
        
        x_[nI] = x_[nI-1] + dx
        y_[nJ] = y_[nJ-1] + dy
        z_[nK] = z_[nK-1] + dz
        
        for n in range(nK+1):
            for m in range(nJ+1):
               for l in range(nI+1):
                   # index current vertex
                   idx = b*(nI+1)*(nJ+1)*(nK+1) + n*(nI+1)*(nJ+1) + m*(nI+1) + l   
                   # x,y,z of cell vertex
                   xcell[idx] = x_[l]              
                   ycell[idx] = y_[m]                 
                   zcell[idx] = z_[n]   
    
    return xcell, ycell, zcell

def find_tree_node(batsrus, point):

    xin = batsrus.xGlobalMin <= point[0] <= batsrus.xGlobalMax
    yin = batsrus.yGlobalMin <= point[1] <= batsrus.yGlobalMax
    zin = batsrus.zGlobalMin <= point[2] <= batsrus.zGlobalMax
    if not (xin and yin and zin): 
        raise RuntimeError('point out of simulation volume')

    found = False
    for iNode in batsrus.amr_level_0_nodes:
        minx = batsrus.block_x_min[F2P(iNode)]
        maxx = batsrus.block_x_max[F2P(iNode)]
        miny = batsrus.block_y_min[F2P(iNode)]
        maxy = batsrus.block_y_max[F2P(iNode)]
        minz = batsrus.block_z_min[F2P(iNode)]
        maxz = batsrus.block_z_max[F2P(iNode)]

        p1 = minx <= point[0] <= maxx
        p2 = miny <= point[1] <= maxy
        p3 = minz <= point[2] <= maxz
        if p1 and p2 and p3:
            found = True
            break

    assert(found == True)

    while True:
        if batsrus.block_child_count[F2P(iNode)] == 0:
            break

        found = False
        for j in range(batsrus.block_child_count[F2P(iNode)]):
            child = batsrus.block_child_ids[j, F2P(iNode)]

            xin = batsrus.block_x_min[F2P(child)] <= point[0] <= batsrus.block_x_max[F2P(child)]
            yin = batsrus.block_y_min[F2P(child)] <= point[1] <= batsrus.block_y_max[F2P(child)]
            zin = batsrus.block_z_min[F2P(child)] <= point[2] <= batsrus.block_z_max[F2P(child)]

            if xin and yin and zin:
                found = True
                iNode = child
                break
 
    return iNode

def F2P(fortran_index):
    return fortran_index - 1

@numba.jit(nopython=True)
def P2F(python_index):
    return python_index + 1

###########################################################
# Code below borrows from swmfio
###########################################################

def get_batsrus_data_from_cdf(file, info):
    """Read BATSRUS data from CDF file.  Store the data in BATSRUS
    following the pattern used by swmfio for BATSRUS
     
    Inputs:
        file = path to CDF file
        
        info = information on MHD data, standard data used throughout code
         
    Outputs:
        Returns batsdata with BATSRUS data
    """
    logging.info('Read BATSRUS file and convert to BATSRUSData')
    
    # Read the file
    cdf = cdfread.CDF(file)
    globatts = cdf.globalattsget()
    time = get_mhd_file_time(file)
    assert( time != -1 )  # Time not found

    npts = int(globatts['number_of_cells'])
    nBlock = int(globatts['number_of_blocks'])
    nI = int(globatts['special_parameter_NX'])
    nJ = int(globatts['special_parameter_NY'])
    nK = int(globatts['special_parameter_NZ'])
    assert( nBlock*nI*nJ*nK == npts )
    
    # Some CDF files have rCurrents, some do not
    if 'r_currents' in globatts:
        rCurrents = np.float64(globatts['r_currents'])
    else:
        rCurrents = info['rCurrents']

    logging.info(f"npts = {npts}")
    logging.info(f"nBlock = {nBlock}")
    logging.info(f"nI/nJ/nK = {nI}/{nJ}/{nK}")
    
    nNode = cdf.varget('block_amr_levels').size
    block2node = -np.ones((nBlock,), dtype=np.int64)
    node2block = -np.ones((nNode,), dtype=np.int64)
    
    logging.info(f"nNode = {nNode}")
    
    varidx = numba.typed.Dict.empty(key_type=numba.types.unicode_type, 
                                    value_type=numba.types.int64,)
    units  = numba.typed.Dict.empty(key_type=numba.types.unicode_type, 
                                    value_type=numba.types.unicode_type,)
    
    nVar = 0
    for cdfvar in cdf.cdf_info()['zVariables']:
        if cdf.varget(cdfvar).shape == (1, npts):
            nVar += 1
    
    nVar += 1 # for added measure (volume) variable
    
    data_arr = np.empty((npts, nVar));
    data_arr[:,:] = np.nan
    
    iVar = 0
    for cdfvar in cdf.cdf_info()['zVariables']:
        try:
            var = cdf.varattsget(cdfvar)['Original Name']
        except:
            var = cdfvar
    
        if cdf.varget(cdfvar).shape == (1, npts):
            data_arr[:, iVar] = cdf.varget(cdfvar)[0,:]
            units[var] = cdf.varattsget(cdfvar)['units']
            varidx[var] = iVar
            iVar += 1
    
    assert(not np.isfortran(data_arr))
    
    if USE_FALSEB:
        logging.info('WARNING: USE_FALSEB is True, fake B field in use. Check options')
        
        # Create a magnetic field due to a line current parallel to x-axis
        # offset 2*yGlobalMax in y-direction (So curl and div are zero inside volume)
        # yGlobalMax = globatts['global_y_max']
        yGlobalMax = 128.0  # Make it match value in SWMF file
        
        # rho squared around x-axis
        rho2 = ( data_arr[:, varidx['y']] + 2*yGlobalMax )**2 + data_arr[:, varidx['z']]**2
        
        # New magnetic field
        data_arr[:, varidx['bx']] = 0.
        data_arr[:, varidx['by']] = - data_arr[:, varidx['z']] / rho2 # by = - sin(phi)/rho
        data_arr[:, varidx['bz']] = + (data_arr[:, varidx['y']] + 2*yGlobalMax ) / rho2 # bz = cos(phi)/rho

        # New field mean magnitude
        Bnew = np.mean( np.sqrt(data_arr[:, varidx['bx']]**2 
                                + data_arr[:, varidx['by']]**2 
                                + data_arr[:, varidx['bz']]**2) )

        # Normalize field to have a mean magnitude of Bmag
        Bmag = 20.0
        data_arr[:, varidx['bx']] = data_arr[:, varidx['bx']] * Bmag / Bnew
        data_arr[:, varidx['by']] = data_arr[:, varidx['by']] * Bmag / Bnew
        data_arr[:, varidx['bz']] = 0.

    if USE_CURLB or USE_FALSEB:
        logging.info('WARNING: USE_CURLB is True, check options')
        # Use curlB to determine current density, j, rather than use OpenGGCM 
        # provided values
        
        DataArray_tmp = data_arr.transpose()
        assert(np.isfortran(DataArray_tmp))
        
        DataArray_tmp = DataArray_tmp.reshape((nVar, nI, nJ, nK,nBlock), order='F')
        assert(np.isfortran(DataArray_tmp))

        data_arr = BATSRUS_curlBtoJ(data_arr, DataArray_tmp, varidx, nVar, nI, nJ, nK, nBlock, rCurrents)

    DataArray = data_arr.transpose()
    assert(np.isfortran(DataArray))
    
    DataArray = DataArray.reshape((nVar, nI, nJ, nK, nBlock), order='F')
    assert(np.isfortran(DataArray))
    
    # Initialize measures
    varidx['measure'] = iVar
    iVar += 1
    for i in range(nBlock):
        dx = DataArray[varidx['x'], 1,0,0, i] -  DataArray[varidx['x'], 0,0,0, i]
        dy = DataArray[varidx['y'], 0,1,0, i] -  DataArray[varidx['y'], 0,0,0, i]
        dz = DataArray[varidx['z'], 0,0,1, i] -  DataArray[varidx['z'], 0,0,0, i]
        DataArray[varidx['measure'], :, :, :, i] = dx*dy*dz

    # Initialize block structure
    block_child_ids = np.array([
                                    cdf.varget('block_child_id_1')[0,:],
                                    cdf.varget('block_child_id_2')[0,:],
                                    cdf.varget('block_child_id_3')[0,:],
                                    cdf.varget('block_child_id_4')[0,:],
                                    cdf.varget('block_child_id_5')[0,:],
                                    cdf.varget('block_child_id_6')[0,:],
                                    cdf.varget('block_child_id_7')[0,:],
                                    cdf.varget('block_child_id_8')[0,:],
                                ])
    block_child_ids = np.array(P2F(block_child_ids))
    
    amr_level_0_nodes = P2F( cdf.varget('block_at_amr_level')[0,:] )
    amr_level_0_nodes = np.array(amr_level_0_nodes)
    
    # Get cells, each cell has an x,y,z point at the center and has
    # dimensions dx * dy * dz with volume measure
    xcell, ycell, zcell = get_batsrus_grid_sub( DataArray, varidx, nBlock, nI, nJ, nK)
    cellvertices = np.column_stack((xcell, ycell, zcell))
    
    # Initial setup of BATSRUSdata, everything but block2node and node2block
    # which are found below.
    batsdata = BATSRUSdata(
                      model             = 'BATSRUS',
                      nDim              = globatts['grid_system_1_number_of_dimensions'],
                      nBlock            = nBlock,
                      nI                = nI,
                      nJ                = nJ,
                      nK                = nK,
                      xGlobalMin        = globatts['global_x_min'],
                      yGlobalMin        = globatts['global_y_min'],
                      zGlobalMin        = globatts['global_z_min'],
                      xGlobalMax        = globatts['global_x_max'],
                      yGlobalMax        = globatts['global_y_max'],
                      zGlobalMax        = globatts['global_z_max'],
                      rCurrents         = rCurrents,
    
                      amr_level_0_nodes = amr_level_0_nodes,
                      block_parent_id   = cdf.varget('block_parent_id')[0,:],
                      block_child_ids   = block_child_ids,
                      block_amr_levels  = np.array(cdf.varget('block_amr_levels')[0,:]),
                      block_x_min       = cdf.varget('block_x_min')[0,:],
                      block_y_min       = cdf.varget('block_y_min')[0,:],
                      block_z_min       = cdf.varget('block_z_min')[0,:],
                      block_x_max       = cdf.varget('block_x_max')[0,:],
                      block_y_max       = cdf.varget('block_y_max')[0,:],
                      block_z_max       = cdf.varget('block_z_max')[0,:],
                      block_child_count = np.array(cdf.varget('block_child_count')[0,:]),
                      
                      cellvertices      = cellvertices ,
     
                      data_arr          = data_arr     ,
                      DataArray         = DataArray    ,
                      varidx            = varidx       ,
    
                      block2node        = block2node   ,
                      node2block        = node2block   ,
                      
                      units             = units         ,
                      time              = np.array(time), 
                      file              = file
                )
    
    # Determine block2node and node2block
    for iBlockP in range(batsdata.block2node.size):
        iNodeP = F2P( find_tree_node(batsdata, data_arr[iBlockP*nI*nJ*nK, 0:3]) )
        batsdata.block2node[iBlockP] = iNodeP
        batsdata.node2block[iNodeP] = iBlockP

    return batsdata
  
if __name__ == "__main__":
    file = '/Volumes/PhysicsHD/Bob_Weigel_070323_3/GM_CDF/3d__ful_4_e20000101-194800-000.out.cdf'
    dir_derived = '/Volumes/PhysicsHD/Bob_Weigel_070323_3.derived'
    info = {} # empty info dict as placeholder
    
    from datetime import datetime
    now = datetime.now()
    print('Start: ', now.time())
    
    batsdata = get_batsrus_data_from_cdf(file,info)
    
    end = datetime.now()
    print('Finish: ', end.time())
    
    from deltaB import convert_mhd_to_dataframe, create_deltaB_spherical_dataframe
    
    df = convert_mhd_to_dataframe( batsdata )
    df = create_deltaB_spherical_dataframe( df )
    
    from deltaB.BATSRUS_to_VTK import BATSRUS_to_VTK

    tovtk = BATSRUS_to_VTK(batsdata)
    tovtk.convert_to_vtk()
    
    import os.path
    basename = os.path.basename(file)
    
    tovtk.write_vtk_to_file( dir_derived, basename, 'vtk')
    
    complete = datetime.now()
    print('Complete: ', complete.time())
   
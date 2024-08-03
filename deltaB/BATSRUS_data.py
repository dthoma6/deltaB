#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:29:07 2024

@author: Dean Thomas
"""

from numba.types import namedtuple

BATSRUSdata = namedtuple('BATSRUSdata', 
                    ['model',
                    'nDim',
                    'nBlock'    ,
                    'nI'        ,
                    'nJ'        ,
                    'nK'        ,
                    'xGlobalMin',
                    'yGlobalMin',
                    'zGlobalMin',
                    'xGlobalMax',
                    'yGlobalMax',
                    'zGlobalMax',
                    'rCurrents' ,

                    'amr_level_0_nodes',
                    'block_parent_id'  ,
                    'block_child_ids'  ,
                    'block_amr_levels' ,
                    'block_x_min'      ,
                    'block_y_min'      ,
                    'block_z_min'      ,
                    'block_x_max'      ,
                    'block_y_max'      ,
                    'block_z_max'      ,
                    'block_child_count',

                    'cellvertices',

                    'data_arr' ,
                    'DataArray',
                    'varidx'   ,

                    'block2node',
                    'node2block',
                    
                    'units',
                    'time' ,
                    'file'])

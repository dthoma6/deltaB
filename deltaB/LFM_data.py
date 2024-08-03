#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:32:55 2024

@author: Dean Thomas
"""

from numba.types import namedtuple

LFMdata = namedtuple('LFMdata', 
                    ['model',
                    'nI'        ,
                    'nJ'        ,
                    'nK'        ,
                    
                    'xGlobalMinSM',    # Min/max in SM coordinates
                    'yGlobalMinSM',
                    'zGlobalMinSM',
                    'xGlobalMaxSM',
                    'yGlobalMaxSM',
                    'zGlobalMaxSM',
                    
                    'rCurrents' ,      # scalar, coordinate independent
                    
                    'data_arr'  ,      # data in GSM coordinates
                    'DataArray' ,      # data in GSM coordinates
                    'varidx'    ,

                    'cellcentersSM'   , # cell center coordinates in SM
                    'cellcentersGSM'  , # cell center coordinates in GSM

                    'cellverticesSM' ,  # vertices in SM coordinates
                    'cellverticesGSM',  # vertices in GSE coordinates

                    'xcenterSM'    ,    # cylindrical coordinates SM
                    'rcenterSM'    ,    # x, r, and az of cell centers
                    'acenterSM'    ,

                    'SM_to_GSM' ,       # Transformation matrices to and from
                    'GSM_to_SM' ,       # SM and GSM

                    'units'     ,
                    'time'      ,
                    'file'])


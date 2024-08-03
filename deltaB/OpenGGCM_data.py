#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:32:55 2024

@author: Dean Thomas
"""

from numba.types import namedtuple

OpenGGCMdata = namedtuple('OpenGGCMdata', 
                    ['model',
                    'nI'        ,
                    'nJ'        ,
                    'nK'        ,
                    
                    'xGlobalMinGSE',    # Min/max in GSE coordinates
                    'yGlobalMinGSE',
                    'zGlobalMinGSE',
                    'xGlobalMaxGSE',
                    'yGlobalMaxGSE',
                    'zGlobalMaxGSE',
                    
                    'rCurrents' ,       # scalar, coordinate independent
                    
                    'data_arr'  ,       # data in GSM coordinates
                    'DataArray' ,       # data in GSM coordinates
                    'varidx'    ,
 
                    'xtickGSE'  ,       # axes Ticks in coordinates in GSE
                    'ytickGSE'  ,
                    'ztickGSE'  ,
                    
                    'cellcentersGSE'  , # cell center coordinates in GSE
                    'cellcentersGSM'  , # cell center coordinates in GSM
                    
                    'cellverticesGSE',  # vertices in GSE coordinates
                    'cellverticesGSM',  # vertices in GSE coordinates
    
                    'GSE_to_GSM',       # matrix to transform to GSM coordinates
                    'GSM_to_GSE',       # matrix to transform to GSM coordinates

                    'units'     ,
                    'time'      ,
                    'file'])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:13:30 2024

@author: Dean Thomas
"""

import pandas as pd

stationfile = './deltaB/stations.txt'
stationdf = pd.read_csv(stationfile, sep='|', header=0)

from collections import namedtuple
Magnetometer = namedtuple('Magnetometer', ['name','csys','ctype','coords'])

print( 'from collections import namedtuple' )
print( "Magnetometer = namedtuple('Magnetometer', ['name','csys','ctype','coords']) ")
print()
print( 'specified_magnetometers = {' )
for index, row in stationdf.iterrows():
    point = row[' Station ']
    lat   = row[' Latitude ']
    long  = row[' Longitude ']

    print( "'" + point + "': Magnetometer(name='" + point + "', csys='GEO', ctype='sph', coords=(1.," + str(lat) + "," + str(long) +") )," )

print('}')
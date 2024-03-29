#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:59:10 2024

@author: Dean Thomas
"""

from CARR_Scenarios1_info import info as info

import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

############################################################################
#
# Script to generate plots of solar wind inputs from CCMC website
#
############################################################################

# Contains the solar wind data from CCMC
swtxt = os.path.join(info["dir_run"], "CARR_Scenario1_IMF.txt")

df = pd.read_csv(swtxt, sep=" ")

# The file "Input Data File Format | CCMC.pdf" gives us the format of the
# solar wind data
df.columns =[ 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'ms', r'$B_x$ (nT)', 
             r'$B_y$ (nT)', r'$B_z$ (nT)', r'$V_x$ (km/s)', r'$V_y$ (km/s)',
             r'$V_z$ (km/s)', r'$N$ (${cm}^{-3}$)', r'$T$ (Kelvin)' ]


df['Datetime'] = pd.to_datetime( df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'ms']] )

DELTAHR = 0.
df['Datetime + Delta'] = df['Datetime'] + timedelta(hours=DELTAHR)

# xlimits =( datetime(2009,9,2,2,9,0,0), datetime(2009,9,2,9,30,0,0) )

# Set some plot configs
plt.rcParams["figure.figsize"] = [12.0,12.0] 
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.dpi"] = 600
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 22
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{cmbright}'

fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)

df.plot( x='Datetime + Delta', y=[r'$N$ (${cm}^{-3}$)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$N$ ({cm}\textsuperscript{-3})', \
                style=['-','r:','--'], \
                grid = False,\
                legend=False,
                ax=ax[0])  
    
df.plot( x='Datetime + Delta', y=[r'$T$ (Kelvin)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$T$ (Kelvin)', \
                style=['-','r:','--'], \
                grid = False,\
                legend=False,
                ax=ax[1])  
    
df.plot( x='Datetime + Delta', y=[r'$V_x$ (km/s)', r'$V_y$ (km/s)', r'$V_z$ (km/s)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$V$ (km/s)', \
                style=['-','r:','--', '.'], \
                grid = False,\
                legend=True,
                ax=ax[2])  

df.plot( x='Datetime + Delta', y=[r'$B_x$ (nT)', r'$B_y$ (nT)', r'$B_z$ (nT)'], \
                xlabel=r'Time (UTC)', \
                ylabel=r'$B$ (nT)', \
                style=['-','r:','--', '.'], \
                grid = False,\
                legend=True,
                ax=ax[3])  

plt.tight_layout()

fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.png" ) )
# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.pdf" ) )
# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.eps" ) )
# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.jpg" ) )
# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.tif" ) )
# fig.savefig( os.path.join( info['dir_plots'], "solar_wind_inputs.svg" ) )
  
    
    
    

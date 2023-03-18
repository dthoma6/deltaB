This repository includes tools to analyze delta B contributions to the earth's total magnetic field.  This version, 0.9.0,  has the ability to determine delta B contributions from the magnetosphere, the gap region, and the ionosphere.  The delta B contributions are determined from the current systems provided by SWMF and the Biot-Savart Law.

The major components are:

1. process_ms.py, process_gap.py, and process_iono.py that use Biot-Savart to determine the magnetic field (in North-East-Down coordinates) at a specified point X.  Biot-Savart calculation uses magnetosphere, gap region, ionosphere current density as appropriate. The current densities are specfied in the applicable SWMF output file for a given time. 

2. plotsHeatmap_Bn.py that loops through data in SWMF (i.e., BAT-S-RUS or RIM files) to create data for heat maps showing the breakdown of Bnorth due to currents parallel and perpendicular to B field at a specified time.  

3. plots2D_Bn.py that loops through data in SWMF files to generate data for 2D plots showing Bnorth versus time including the breakdown of contributions from currents parallel and perpendicular to local B field.  

4. plots2D_BATSRUS that generates various 2-D plots of various parameters in BATS-R-US files. For example how current density or velocity vary with range from earth.  

5. BATSRUS_dataframe that creates a Pandas dataframe from information within a BATSRUS file and calculates various quantities such as the delta B contribution from each BATSRUS grid cell.

6. coordinates.py and util.py that provide cooridnate transformation and utility functions. 

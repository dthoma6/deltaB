This repository includes tools to analyze $\delta B$ contributions from the magnetosphere, 
ionosphere, and gap-region to the magnetic field measured on Earth.  That is, the change in
the magnetic field observed on earth due to magnetospheric, gap-region and iononspheric 
currents.  This version, 1.0.0, has the ability to determine contributions from Space Weather 
Modeling Framework (SWMF) current densities.

# Install

```
git clone https://github.com/dthoma6/deltaB/tree/Carrington-Event-Paper
cd deltaB
pip install --editable .
```

# Carrington Event Paper

The scripts that generate the plots used in the Carrington Event paper
are in the "runs" directory.  One directory for the Scenario 1 (CARR_Scenario1) and
one directory for Scenario 2 (Chigomezyo_Ngwira_092112_3a).  **Each directory contains
a README which tells the user how to download the data to be analyzed and how to 
execute the scripts.**  

The scripts will create new directories for the results.  The plots will stored 
in two new directories, CARR_Scenario1.plots and Chigomezyo_Ngwira_092112_3a.plots. 
Other data that the scripts generate will be stored in CARR_Scenario1.derived and 
Chigomezyo_Ngwira_092112_3a.derived

# Tests

The "test" directory contains tests to execute on the installed package.

1. Comparison_to_CCMC compares the results from the deltaB package to results
   from CCMC and from magnetopost.  Results.txt provides the numeric test results
   and result_plots are the plots.  Your results can be compared to these
   results and the plots to ensure consistency.  The data directory contains
   the CCMC and magnetopost results that deltaB is compared against.
2. compare_Weigel_data.py compares a Paraview calculation of magnetosphere $\delta B$
   to deltaB's calculation.
3. calc_gap_b_test compares deltaB calculation of gap-region $\delta B$ to an
   analytic solution for a simple scenario involving a line current through the
   north pole. 

# Description

In processing SWMF results, it uses both the magnetosphere and ionosphere files 
to determine $\delta B$ contributions to the magnetic field at a specified point. 
The user can specify a point on the Earth's surface, for example, a magnetometer 
site. Or the user can specify a point in space.  The algorithms will determine 
the contributions that the magnetosphere, ionosphere, and gap-region make to the 
$B$ field in North-East-Down components, dividing the contributions into 
various catergories.  For example, it determines contributions from currents 
parallel and perpendicular to the local $B$ field, ionospheric Hall and Pedersen 
currents, and Field-Aligned Currents in the SWMF gap-region.  It also considers
specific geospace regions, that is, $\delta B$ contributions from the magnetosheath,
the near-Earth region, and the neutral sheet region.

The major components are:

1. process_ms.py, process_gap.py, and process_iono.py uses Biot-Savart to 
determine the magnetic field (in North-East-Down coordinates) at a user-specified 
point.  The Biot-Savart calculations use the magnetosphere, gap-region, ionosphere 
current densities as appropriate. The current densities are specfied in the 
applicable SWMF output file. 

2. plots2D_Bned.py loops through data in SWMF files to generate 
2D plots showing the evolution of $B_{north}$, $B_{east}$ and $B_{down}$ versus 
time.  

3. plots2D_Bn.py loops through data in SWMF files to generate 2D 
plots showing $B_{north}$ versus time.  This procedure differs from plots2D_Bned.py,
in that it focuses on $B_{north}$ rather than all three components.  And it 
further divides magnetospheric currents into those parallel and perpendicular to 
local $B$ field.  

5. plots2D_BATSRUS.py and plot2D_BATSRUSparams.py generate 2D plots of 
various parameters in the SWMF BATSRUS files. For example, how current density or 
velocity vary with range from earth.  

6. plotsHeatmapWorld_Bn.py loops through data in SWMF files (i.e., BATSRUS or 
RIM files) to generate heat maps showing the breakdown of $B_{north}$ due to specific
currents and geospace regions.  

7. BATSRUS_dataframe.py creates a Pandas dataframe from information within a 
BATSRUS file and calculates various quantities such as the $\delta B$ contribution 
from each BATSRUS grid cell.  process_ms.py calls this procedure.

8. find_boundaries.py determines the locations of the bow shock, 
magnetopause, and neutral sheet based on standard boundary conditions.

9. deltaB_by_region.py uses Biot-Savart to determine the magnetic field 
(in North-East-Down coordinates) at a specified point.  Biot-Savart calculation 
uses magnetosphere current density and the locations of the bow shock, magnetopause, 
and neutral sheet (see find_boundaries.py) to determine the contributions from 
specific geospace regions.

10. plotting.py, coordinates.py and util.py provide plotting, cooridnate 
transformation, and utility functions. 


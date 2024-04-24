
The BATSRUS_interpolator uses the CCMC OCTREE library. From Kamodo, download 
the OCTREE library:

https://github.com/nasa/Kamodo/tree/master/kamodo_ccmc/readers/OCTREE_BLOCK_GRID

The OCTREE files must be located with the other deltaB Python files.  So create
{Python Code}/deltaB/deltaB/OCTREE_BLOCK_GRID, where {Python Code}/deltaB/deltaB 
contains the deltaB Python files.  The directory {Python Code}/deltaB/deltaB 
should contain:

BATSRUS_dataframe.py		plots2D_Bned.py
BATSRUS_interpolator.py		plotsHeatmapWorld_Bn.py
OCTREE_BLOCK_GRID		plotting.py
__init__.py			process_gap.py
coordinates.py			process_ms.py
deltaB_by_region.py		process_ms_divBint.py
find_boundaries.py		process_ms_surfint_outer.py
magnetometers.py		process_ms_surfint_rCurrents.py
plots2D_BATSRUS.py		stations.txt
plots2D_BATSRUSparams.py	untitled0.py
plots2D_Bn.py			util.py

To build the CCMC OCTREE library, from the terminal, change into the 
OCTREE_BLOCK_GRID directory, and execute:

python interpolate_amrdata_extension_build.py







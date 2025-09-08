
This variant support $\delta B$ analysis of SWMF, LFM, and OpenGGCM simulations.
The SWMF and OpenGGCM code has been tested.  The LFM code is not fully tested.

# Install

```
git clone https://github.com/dthoma6/deltaB/tree/surface_integral
cd deltaB
pip install --editable .
```

# Description

This repository includes tools to analyze $\delta B$ contributions from the magnetosphere, 
ionosphere, and gap-region to the magnetic field on Earth.  This version uses a
the BATSRUS_interpolator from the CCMC OCTREE library. From Kamodo, download 
the OCTREE library:

https://github.com/nasa/Kamodo/tree/master/kamodo_ccmc/readers/OCTREE_BLOCK_GRID

The OCTREE files must be located with the other deltaB Python files.  So create
{Python Code}/deltaB/deltaB/OCTREE_BLOCK_GRID, where {Python Code}/deltaB/deltaB 
contains the deltaB Python files.  Among other files, the directory {Python Code}/deltaB/deltaB 
should contain:

```
BATSRUS_dataframe.py		plots2D_Bned.py
BATSRUS_interpolator.py plotsHeatmapWorld_Bn.py
OCTREE_BLOCK_GRID		plotting.py
__init__.py			    process_gap.py
coordinates.py			process_ms.py
deltaB_by_region.py		process_ms_divBint.py
find_boundaries.py		process_ms_surfint_outer.py
magnetometers.py		    process_ms_surfint_rCurrents.py
plots2D_BATSRUS.py		stations.txt
plots2D_BATSRUSparams.py util.py
plots2D_Bn.py			(Plus other files)
```

To build the CCMC OCTREE library, from the terminal, change into the 
OCTREE_BLOCK_GRID directory, and execute:

```
python interpolate_amrdata_extension_build.py
```

In processing SWMF, OpenGGCM, or LFM results, the code uses the magnetosphere files 
to determine $\delta B$ contributions to the magnetic field at a specified point. 
The user can specify a point on the Earth's surface, for example, a magnetometer 
site. Or the user can specify a point in space.  The algorithms will determine 
the contributions that the magnetosphere makes to the $B$ field in North-East-Down components.

The key difference between this branch and earlier branches are of this code are in
two areas. First, the code was expanded beyond analysis of SWMF code to include 
OpenGGCM (tested) and LFM (partially tested) magnetosphere results.  Second, it allows an 
examination of the results using the Helmholtz decomposition theorem.  

We examine the size of $\nabla \cdot B$ and outer surface boundary integrals 
in estimating the surface magnetic field from 
magnetohydrodynamic (MHD) simulations. Maxwell’s equations tell us 
$\nabla \cdot B = 0$, which may be violated due to numerical error. MHD models 
such as the SWMF and the OpenGGCM use different techniques to limit $\nabla \cdot B$. 
Analyses of MHD simulations typically assume $\nabla \cdot B$ errors are small. 
Similarly, analyses commonly use the Biot-Savart Law and magnetospheric current 
density estimates from MHD simulations to determine the magnetic field at a 
specific point on Earth. This calculation frequently omits the surface integral 
over the outer boundary of the simulation volume that the Helmholtz decomposition 
theorem requires. This code uses SWMF and OpenGGCM simulations to estimate the 
magnitudes of the $\nabla \cdot B$and outer boundary integrals compared to 
Biot-Savart estimates of the magnetic field on Earth. 

The major components are:

1. process_ms.py uses Biot-Savart to determine the magnetic field 
(in North-East-Down coordinates) at a user-specified 
point.  The Biot-Savart calculations use the magnetosphere, gap-region, ionosphere 
current densities as appropriate. The current densities are specfied in the 
applicable MHD output file.  This component largely reuses the code from the earlier
branches.

2. process_ms_divBint.py, process_ms_surfint_outer.py, and process_ms_surfint_rCurrents.py
calculate the divergence of B volume integral, the outer surface integral, and
the inner (rCurrents) surface integrals that are terms in the Helmholtz decomposition 
theorem. 


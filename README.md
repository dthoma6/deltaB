
This branch supports $\delta B$ analysis of SWMF, LFM, and OpenGGCM simulations.
The SWMF and OpenGGCM code have been tested.  The LFM code is not fully developed 
or tested.

# Install

```
git clone https://github.com/dthoma6/deltaB/tree/surface_integral
cd deltaB
pip install --editable .
```

# Kamodo Requirements

This branch includes tools to analyze $\delta B$ contributions from the magnetosphere 
to the magnetic field on Earth.  This version uses a the BATSRUS_interpolator 
from the CCMC OCTREE library. From Kamodo, download the OCTREE library:

https://github.com/nasa/Kamodo/tree/master/kamodo_ccmc/readers/OCTREE_BLOCK_GRID

The OCTREE files must be located with the other deltaB Python files.  So create
{Python Code}/deltaB/deltaB/OCTREE_BLOCK_GRID, where {Python Code}/deltaB/deltaB 
contains the deltaB Python files.  Among other files, the directory 
{Python Code}/deltaB/deltaB should contain:

```
__init__.py
BATSRUS_curlB.py
BATSRUS_data.py
BATSRUS_dataframe.py
BATSRUS_divBint_b.py
BATSRUS_interpolator.py
BATSRUS_interpolator2.py
BATSRUS_surfint_outer_b.py
...
OCTREE_BLOCK_GRID
OpenGGCM_curlB.py
OpenGGCM_data.py
OpenGGCM_dataframe.py
OpenGGCM_divBint_b.py
OpenGGCM_interpolator.py
OpenGGCM_interpolator2.py
...
```

To build the CCMC OCTREE library, from the terminal, change into the 
OCTREE_BLOCK_GRID directory, and execute:

```
python interpolate_amrdata_extension_build.py
```

# Description

In processing SWMF, OpenGGCM, or LFM results, the code uses the magnetosphere, 
gap region, and ionosphere files to determine $\delta B$ contributions to the 
magnetic field at a specified point. The user can specify a point on the Earth's 
surface, for example, a magnetometer site. Or the user can specify a point in 
space.  The algorithms will determine the contributions that the magnetospheric,
ionospheric, and gap region current densities makes to the $B$ field in 
North-East-Down components.

The key difference between this branch and earlier branches are in
two areas. First, the code was expanded beyond analysis of SWMF code to include 
OpenGGCM (tested) and LFM (partially tested) results.  Second, it allows an 
examination of the results using the Helmholtz decomposition theorem, which states 
states that a vector field, in this case the magnetic field $B$, can be written 
as the sum of irrotational and solenoidal components. This results in a volume 
integral involving $\nabla \times B$, a volume integral involving $\nabla \cdot B$, 
and surface integrals over the boundary of the volume.

With this code, we examine the size of $\nabla \cdot B$ and outer surface boundary 
integrals in estimating the surface magnetic field from 
magnetohydrodynamic (MHD) simulations. Maxwell’s equations tell us 
$\nabla \cdot B = 0$, which may be violated due to numerical error. Various 
MHD models use different techniques to limit $\nabla \cdot B$. 
Analyses of MHD simulations typically assume $\nabla \cdot B$ errors are small. 
Similarly, analyses commonly use the Biot-Savart Law and magnetospheric current 
density estimates from MHD simulations to determine the magnetic field at a 
specific point on Earth. This calculation frequently omits the surface integral 
over the outer boundary of the simulation volume that the Helmholtz decomposition 
theorem requires. This code uses MHD simulation results to estimate the 
magnitudes of the $\nabla \cdot B$ and outer boundary integrals compared to 
Biot-Savart estimates of the magnetic field on Earth. 

For examining results from magnetospheric contributions, the major components are:

1. process_ms.py uses Biot-Savart to determine the magnetic field 
(in North-East-Down coordinates) at a user-specified point.  The Biot-Savart 
calculations use the magnetosphere current densities, which are specfied in the 
applicable MHD output file.  This component largely reuses the code from the earlier
branches.

2. process_ms_divBint.py, process_ms_surfint_outer.py, and process_ms_surfint_rCurrents.py
calculate the $\nabla \cdot B$ volume integral, the outer boundary surface integral, 
and the inner (rCurrents) boundary surface integrals that are terms from the 
Helmholtz decomposition theorem. 


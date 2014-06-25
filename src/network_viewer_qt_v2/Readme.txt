1) If you have OpenCL libraries (or dll's) and headers installed 
and you want to use OpenCL-based rendering in Network Viewer 2, 
define the variables 'OPENCL_INCL' and 'OPENCL_LIB_FLD' in the file 
'../../config'.
  If your video card does not support OpenCL, Network Viewer 2 
will be compiled successfully but will show an error and will ask 
about usage of OpenMP-based renderer. OpenMP support is not required 
(I hope!) for OpenMP-based renderer.
  Anyway, 2D renderer works separately from 3D renderer.

2) You must not include the file 'core/renderer_OpenCL.ocl' in any 
projects for compilation and/or linkage. This file is used only by 
'core/renderer_OpenCL.cpp'.
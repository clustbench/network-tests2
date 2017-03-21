network-tests2 
==============

Benchmarks and analysis of interconnection in HPC cluster

Comiple dependencies and process
--------------------

This software has folowing dependencies to compile it:

1. Requires
    1. C,C++ compilers: for example gcc,g++ 
    2. autoconf(2.69), automake(1.14)
    3. GnuMake
    4. NetCDF library with C aplication interface for it
2. For benchmarking requires:
    1. MPI-2 standard implementation: OpenMPI, IntelMPI and so one
3. For GUI requires:
    1. Qt version 4.5 but not Qt5. It requires qmake-qt4 tool.
    2. Qwt5, libqwt5-qt4
    3. OpenGL
    4. Support OpenMP in compiler and optionally OpenCL
4. For clustering requires
    1. NetCDF C++ interface.

To compile software you need to proceed folowing steps:

0. Install software dependencies.
1.  Change directory to directory with sources. 
    Then generate configure script by files configure.ac and macroces from 
   ac-macros. This step is performed by runnig make_configure.sh script.
2. Learn fitures and so one in configure file 
     ```
       ./configure --help 
     ```
3. Run ./configure, where show required components and prefix. For example:
    ```
        ./configure --prefix=$HOME/nt-2 --enable-qt-gui
    ```
4. Run make to compile necesossary components
5. Run make install to install all into the prefix directory. Please be carefully
   if prefix in configure was /usr or one of system catalogues. There is no correct
   uninstall script and you have to delete components manually. 


Catalogues structure
--------------------

There are some catalogues:
1. ac-macros - macroces for configure
2. doc - supplementary documentation
3. share - supplementary tools and scripts
4. src/clustering - Clustering tools.
5. src/core - source files requires for many components. 
6. src/network_test - benchmarking aplication
7. src/network_viewer_qt_v2 - tool for drawing results of benchmarking
   and it visual analyzis.
8. java - deprecated java GUI for vizualizing results.    

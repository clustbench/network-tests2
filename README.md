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
    1. Qt version 5.15 and greater
    2. Qwt6 and greater
    3. OpenGL
    4. Support OpenMP in compiler and optionally OpenCL
4. For clustering requires
    1. NetCDF C++ interface

To compile software you need to proceed folowing steps:

0. Install software dependencies.
   If you compile Qwt library from source code, add the following lines in
   the Network_viewer_qt_v2.pro file (in this example the library was installed
   in /usr/local/):
   ```
        INCLUDEPATH += /usr/local/qwt-6.2.0/include
        DEPENDPATH += /usr/local/qwt-6.2.0/include
   ```
   At the end of Network_viewer_qt_v2.pro file add the path of qwt.prf file.
   Example:
   ```
        include ( /usr/local/qwt-6.2.0/features/qwt.prf )
   ```
   Add path to the Qwt library to LD_LIBRARY_PATH variable.
   Example:
   ```
        export LD_LIBRARY_PATH=/usr/local/qwt-6.2.0/lib
   ```
1. Change directory to directory with sources.
   Then generate configure script by files configure.ac and macros from
   ac-macros directory. This step is performed by runnig make_configure.sh
   script.
2. Learn fitures and so one in configure file
     ```
       ./configure --help
     ```
3. Run ./configure, where show required components and prefix. For example:
    ```
        ./configure --prefix=$HOME/nt-2 --enable-qt-gui
    ```
4. Run make to compile necesossary components
5. Run make install to install all into the prefix directory. Please be
   carefully if prefix in configure was /usr or one of system catalogues. There
   is no correct uninstall script and you have to delete components manually.


Catalogues structure
--------------------

There are some catalogues:
1. ac-macros - macros for configure
2. doc - supplementary documentation
3. share - supplementary tools and scripts
4. src/clustering - Clustering tools.
5. src/core - source files requires for many components.
6. src/network_test - benchmarking aplication
7. src/network_viewer_qt_v2 - tool for drawing results of benchmarking
   and it visual analyzis.
8. java - deprecated java GUI for vizualizing results.


References on project
------------------------

Some information on principles of utilities organization can be 
found at this list of articles:
1. Gorelov A., Maysuradze A., Salnikov A. Delay structure mining in computing cluster //
   CEUR Workshop Proceedings. - Vol. 1482. - Aachen : M. Jeusfeld c/o Redaktion Sun SITE, 
   Informatik V, RWTH Aachen Germany Germany, 2015. -  P. 546-551.
2. Bannikov P.S., Salnikov A.N.. Retrieving topology of interconnections 
   in computational cluster based on results of MPI benchmarks. Moscow University 
   Computational Mathematics and Cybernetics. vol. 38, n. 2, pp. 73-82, 2014. 
   DOI: [10.3103/S0278641914020022](http://dx.doi.org/10.3103/S0278641914020022)
3. Salnikov A.N., Andreev D.Yu, Lebedev R.D.. Toolkit for analyzing the 
   communication environment characteristics of a computational cluster 
   based on MPI standard functions. Moscow University Computational Mathematics 
   and Cybernetics. vol. 36, n. 1, pp. 41-49, 2012.
   DOI: [10.3103/S0278641912010074](http://dx.doi.org/10.3103/S0278641912010074)
4. Alexey, S., Dmitry, A., and Roman, L. The analysis of cluster interconnect 
   with the network_tests2 toolkit. In Recent Advances in the Message Passing 
   Interface - 18th European MPI Users' Group Meeting, EuroMPI 2011, Santorini,
   Greece, September 18-21, 2011. Proceedings (Heidelberg, Germany, 2011),
   vol. 6960 of Lecture Notes in Computer Science, Heidelberg, Germany, pp. 160-169.
   DOI: [10.1007/978-3-642-24449-0_19](http://dx.doi.org/10.1007/978-3-642-24449-0_19)
5. Salnikov Alexey N., Andreev Dmitry Y.. An MPI-Based System for Testing
   Multiprocessor and Cluster Communications. Lecture Notes in Computer 
   Science. n. 5205, pp. 332-333, 2008.
   DOI: [10.1007/978-3-540-87475-1_48](http://dx.doi.org/10.1007/978-3-540-87475-1_48)


Deployment on Clusters
----------------------

You can learn how to easily (or not) deploy this tool on HPC clusters
[here](doc/how_to_deploy_on_clusters.md).

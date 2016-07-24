#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([network-tests2], [2.0.1], [Alexey Salnikov <salnikov@cs.msu.ru>])
#AC_CONFIG_SRCDIR([parus_config.h])
#AC_CONFIG_HEADERS([config.h])

AC_ARG_VAR([JAVAC], [Place where javac is installed])
AC_ARG_VAR([JAR], [Place where jar is installed])
AC_ARG_VAR([ANT], [Place where ant utility is installed])

AC_ARG_VAR([MPICC], [Path to MPI2 compatible C-compiler])
AC_ARG_VAR([MPICXX], [Path to MPI2 compatible C++-compiler])
AC_ARG_VAR([MPI_CFLAGS], [Flags for MPI C compiler by default same with CFLAGS])
AC_ARG_VAR([MPI_CXXFLAGS], [Flags for MPI C++ compiler by default same with CXXFLAGS])
AC_ARG_VAR([MPI_LIBS], [Additional libs for parallel program])
AC_ARG_VAR([MPI_LDFLAGS], [Path to additional libs for parallel program])
AC_ARG_VAR([GOALS], [List of goals to be built])

AC_ARG_ENABLE([java-gui],
              [AS_HELP_STRING([--enable-java-gui],
                              [Enable Java version of network-viewer program (disabled by default)]
                             )
              ],
              [AS_CASE(["${enableval}"],
                       [yes], [AS_VAR_APPEND([GOALS], [" java "])],
                       [no],  [],
                       [AC_MSG_ERROR([Bad value ${enableval} for --enable-java-gui])]
                      )
              ]
              # nothing to do because feature is disabled by default
             ) # java-gui

AC_ARG_ENABLE([qt-gui],
              [AS_HELP_STRING([--enable-qt-gui],
                              [Enable Qt version of network-viewer program (disabled by default)]
                             )
              ],
              [AS_CASE(["${enableval}"],
                      [yes], [AS_VAR_APPEND([GOALS], [" network_viewer_qt_v2 "])],
                      [no],  [],
                      [AC_MSG_ERROR([Bad value ${enableval} for --enable-qt-gui])]
                     )
              ]
              # nothing to do because feature is disabled by default
             ) # qt-gui

AC_ARG_ENABLE([network-test],
              [AS_HELP_STRING([--enable-network-test],
                              [Enable testing (enabled by default)]
                             )
              ],
              [AS_CASE(["${enableval}"],
                       [yes], [AS_VAR_APPEND([GOALS], [" network_test "])],
                       [no],  [],
                       [AC_MSG_ERROR([Bad value ${enableval} for --enable-network-test])]
                      )
              ],
              [AS_VAR_APPEND([GOALS], [" network_test "])]
             ) # network-test

AC_ARG_ENABLE([clustering],
              [AS_HELP_STRING([--enable-clustering],
                              [Enable clustering (disabled by default)]
                             )
              ],
              [AS_CASE(["${enableval}"],
                      [yes], [AS_VAR_APPEND([GOALS], [" clustering "])],
                      [no],  [],
                      [AC_MSG_ERROR([Bad value ${enableval} for --enable-qt-gui])]
                     )
              ]
              # nothing to do because feature is disabled by default
             ) # clustering

AS_IF([test "x$JAVAC" = "x"], [AC_SUBST([JAVAC], [javac])])
AS_IF([test "x$JAR" = "x"], [AC_SUBST([JAR], [jar])])
AS_IF([test "x$ANT" = "x"], [AC_SUBST([ANT], [ant])])
AS_IF([test "x$MPICC" = "x"], [AC_SUBST([MPICC], [mpicc])])
AS_IF([test "x$MPICXX" = "x"], [AC_SUBST([MPICXX], [mpicxx])])

AC_PREFIX_DEFAULT([$HOME/network_tests-$PACKAGE_VERSION])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
#AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_RANLIB

AC_CHECK_PROG([HAVE_MPICC], $MPICC, [true], [false])
AS_IF([test "x$HAVE_MPICC" = "xfalse"],
      [AC_MSG_ERROR([MPI compiler for C programming language is not found])]
     )

AC_CHECK_PROG([HAVE_MPICXX], $MPICXX, [true], [false])
AS_IF([test "x$HAVE_MPICXX" = "xfalse"],
      [AC_MSG_ERROR([MPI compiler for C++ programming language is not found])]
     )

AS_IF([test "x$MPI_CFLAGS" = "x"], [AC_SUBST([MPI_CFLAGS], $CFLAGS)])
AS_IF([test "x$MPI_CXXFLAGS" = "x"], [AC_SUBST([MPI_CXXFLAGS], $CXXFLAGS)])
AS_IF([test "x$MPI_LDFLAGS" = "x"], [AC_SUBST([MPI_LDFLAGS], $LDFLAGS)])
AS_IF([test "x$MPI_LIBS" = "x"], [AC_SUBST([MPI_LIBS], $LIBS)])

#AX_MPI([`echo MPI found'],[`echo MPI not FOUND'])
#AC_OPENMP
# Checks for libraries.
# FIXME: Replace `main' with a function in `-lnetcdf':
AC_CHECK_LIB([netcdf], [nc_open])
AC_CHECK_LIB([qwt-qt4],[QwtPlot::axisScaleEngine])
# FIXME: Replace `main' with a function in `-lparus_network':
#AC_CHECK_LIB([parus_network], [main])

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h string.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD
AC_CHECK_FUNCS([clock_gettime floor gethostname gettimeofday localeconv memset sqrt strchr strstr])

AC_CONFIG_FILES([config Makefile parus_config.h])
AC_OUTPUT

#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([network-tests2], [2.0.1], [Alexey Salnikov <salnikov@cs.msu.ru>])

# Wrapper for AC_ARG_ENABLE for basic rules
AC_DEFUN([AX_ARG_ENABLE],
         [AC_ARG_ENABLE([$1],
                        [AS_HELP_STRING([--enable-$1], [$4])],
                        [AS_CASE(["${enableval}"],
                                 [yes], [AS_VAR_APPEND([GOALS], [$2])],
                                 [no],  [],
                                 [AC_MSG_ERROR([Bad value ${enableval} for --enable-$1])]
                                )
                        ]
                        [AS_VAR_APPEND([GOALS], [$3])]
                       )
         ]
        )

# Wrapper for AC_CHECK_PROG with gentle error
AC_DEFUN([AX_CHECK_PROG],
         [AC_CHECK_PROG([HAVE_$1], $$1, [true], [false])
          AS_IF([test "x$HAVE_$1" == "xfalse"],
                [AC_MSG_ERROR([$2])]
               )
         ]
        )

AC_ARG_VAR([JAVAC], [Place where javac is installed])
AC_ARG_VAR([JAR], [Place where jar is installed])
AC_ARG_VAR([ANT], [Place where ant utility is installed])
# TODO: Change the value for qmake for QT5 later
AC_ARG_VAR([QMAKE], [Place where qmake-qt4 is installed])
AC_ARG_VAR([MPICC], [Path to MPI2 compatible C-compiler])
AC_ARG_VAR([MPICXX], [Path to MPI2 compatible C++-compiler])
AC_ARG_VAR([MPI_CFLAGS], [Flags for MPI C compiler by default same with CFLAGS])
AC_ARG_VAR([MPI_CXXFLAGS], [Flags for MPI C++ compiler by default same with CXXFLAGS])
AC_ARG_VAR([MPI_LIBS], [Additional libs for parallel program])
AC_ARG_VAR([MPI_LDFLAGS], [Path to additional libs for parallel program])
AC_ARG_VAR([GOALS], [List of goals to be built])

AX_ARG_ENABLE([java-gui], [" java "], [""],
              [Enable Java version of network-viewer program (disabled by default)]
             ) # java
AX_ARG_ENABLE([qt-gui], [" network_viewer_qt_v2 "], [""],
              [Enable Qt version of network-viewer program (disabled by default)]
             ) # network_viewer_qt_v2
AX_ARG_ENABLE([network-test], [" network_test "], [" network_test "],
              [Enable testing (enabled by default)]
             ) # network-test
AX_ARG_ENABLE([clustering], [" clustering "], [""],
              [Enable clustering (disabled by default)]
             ) # clustering

AC_ARG_ENABLE([all],
              [AS_HELP_STRING([--enable-all], [Enable all modules])],
              [AS_CASE(["${enableval}"],
                       [yes], [AC_SUBST([GOALS], ["java network_test network_viewer_qt_v2 clustering"])],
                       [no],  [],
                       [AC_MSG_ERROR([Bad value ${enableval} for --enable-all])]
                      )
              ]
              # nothing to do because feature is disabled by default
             ) # all

AS_IF([test "x$JAVAC" = "x"], [AC_SUBST([JAVAC], [javac])])
AS_IF([test "x$JAR" = "x"], [AC_SUBST([JAR], [jar])])
AS_IF([test "x$ANT" = "x"], [AC_SUBST([ANT], [ant])])
AS_IF([test "x$MPICC" = "x"], [AC_SUBST([MPICC], [mpicc])])
AS_IF([test "x$MPICXX" = "x"], [AC_SUBST([MPICXX], [mpicxx])])
AS_IF([test "x$QMAKE" = "x"], [AC_SUBST([QMAKE], [qmake-qt4])])

AC_PREFIX_DEFAULT([$HOME/network_tests-$PACKAGE_VERSION])

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_LN_S
AC_PROG_RANLIB

AX_CHECK_PROG([MPICC], [MPI compiler for C programming language has not been found])
AX_CHECK_PROG([MPICXX], [MPI compiler for C++ programming language has not been found])
AS_CASE(["$GOALS"],
        [*"java"*], [
            AX_CHECK_PROG([ANT], [ant is required but has not been found])
            AX_CHECK_PROG([JAR], [jar is required but has not been found])
            AX_CHECK_PROG([JAVAC], [javac is required but has not been found])
            ],
        [*"qt"*], [
            AX_CHECK_PROG([QMAKE], [qmake-qt4 is required but has not been found])
            AC_CHECK_LIB([qwt-qt4],[QwtPlot::axisScaleEngine])
            ]
       )

AS_IF([test "x$MPI_CFLAGS" = "x"], [AC_SUBST([MPI_CFLAGS], $CFLAGS)])
AS_IF([test "x$MPI_CXXFLAGS" = "x"], [AC_SUBST([MPI_CXXFLAGS], $CXXFLAGS)])
AS_IF([test "x$MPI_LDFLAGS" = "x"], [AC_SUBST([MPI_LDFLAGS], $LDFLAGS)])
AS_IF([test "x$MPI_LIBS" = "x"], [AC_SUBST([MPI_LIBS], $LIBS)])

# Checks for libraries.
# -lqwt-qt4 -lnetcdf -lGLU -lQtOpenGL -lQtGui -lQtCore -lGL -lpthread
AC_CHECK_LIB([netcdf], [nc_open])

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
